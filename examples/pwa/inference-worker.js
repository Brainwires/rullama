// rullama inference Dedicated Worker.
//
// Why this exists: the wasm Model lived on the main thread. On iPhone 16e
// (A18, iOS 26, 8 GB shared RAM), the first `step()` would crash the
// WebContent process — a combination of (a) main-thread responsiveness
// reaped by iOS's page watchdog while 35 layers of weight uploads
// streamed in, and (b) ~455 Promise allocations from the async OPFS
// reader piling up in the microtask queue. The reference brainwires-
// chat-pwa project sidesteps both by running its wasm inside a Worker
// and using `FileSystemSyncAccessHandle.read()` (worker-only API).
// This file mirrors that pattern.
//
// Wire protocol (main → worker), one message per RPC:
//   { requestId, type, ...args }
//
//   type values:
//     load          { modelKey, filename, totalBytes, maxContext, textOnly }
//     encode        { text }
//     step          { tokenId }
//     stepWithEmb   { embedding }                # Float32Array
//     tokenStr      { id }
//     isEos         { id }
//     reset         {}
//     setSampling   { opts }
//     renderChat    { messages, withBos }
//     imageSentinelIds {}
//     audioSentinelIds {}
//     imageSoftTokenCount { h, w }
//     decodeWav     { bytes }
//     encodeImage   { pixels, h, w }
//     encodeAudio   { pcm }
//     free          {}
//
// Wire protocol (worker → main), one reply per RPC:
//   { requestId, ok: true,  result }
//   { requestId, ok: false, error }

import init, { Model } from '/pkg/rullama.js';

const OPFS_DIR = 'rullama-models';

// Bridge worker-side console.log (incl. wasm `web_sys::console::log_1`) to
// /api/log so we see what the Rust step path was doing right before any
// iPhone WebContent crash. Wrap, don't replace, so the worker's normal log
// path also still shows in Web Inspector.
const _origLog = console.log.bind(console);
console.log = function (...args) {
    _origLog(...args);
    try {
        const msg = args.map(a => typeof a === 'string' ? a : String(a)).join(' ');
        fetch('/api/log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tag: 'rs', msg, ts: Date.now() }),
            keepalive: true,
        }).catch(()=>{});
    } catch (_) {}
};

// Worker-scope state.
let wasmReady    = null;     // promise that resolves once `init()` has run
let model        = null;     // wasm Model handle
let syncHandle   = null;     // OPFS FileSystemSyncAccessHandle (held for model lifetime)
let totalBytes   = 0;        // GGUF size

function log(...args) {
    const msg = args.map(a => String(a)).join(' ');
    // Main-thread console (visible in Web Inspector).
    self.postMessage({ type: 'log', args: args.map(a => String(a)) });
    // Also POST to /api/log with keepalive so the trace survives a
    // WebContent crash — this is the only way to see what the worker
    // was doing right before iOS Jetsam'd the page on iPhone.
    try {
        fetch('/api/log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tag: 'wkr', msg, ts: Date.now() }),
            keepalive: true,
        }).catch(()=>{});
    } catch (_) {}
}

async function ensureWasm() {
    if (!wasmReady) wasmReady = init();
    return wasmReady;
}

// Open OPFS file by modelKey/filename and return a SYNC read callback
// for the wasm OpfsFetcher. Closes any previously-held handle first.
async function openSyncReadFn(modelKey, filename) {
    if (syncHandle) {
        try { syncHandle.close(); } catch (_) {}
        syncHandle = null;
    }
    const root     = await navigator.storage.getDirectory();
    const dlDir    = await root.getDirectoryHandle(OPFS_DIR, { create: false });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
    const fh       = await modelDir.getFileHandle(filename, { create: false });
    syncHandle = await fh.createSyncAccessHandle();
    const size = syncHandle.getSize();
    if (size === 0) {
        try { syncHandle.close(); } catch (_) {}
        syncHandle = null;
        throw new Error(`OPFS file ${modelKey}/${filename} is empty`);
    }
    const readFn = (offset, length) => {
        // SYNC read — the only reason we're in a Worker. Returns a fresh
        // Uint8Array; wasm copies into linear memory and the buffer is
        // immediately eligible for GC.
        const buf = new Uint8Array(length);
        syncHandle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, totalBytes: size };
}

// === Handlers ===

async function handleLoad({ modelKey, filename, maxContext, textOnly }) {
    await ensureWasm();
    if (model) {
        try { model.free?.(); } catch (_) {}
        model = null;
    }
    const { readFn, totalBytes: size } = await openSyncReadFn(modelKey, filename);
    totalBytes = size;
    log(`load: opening Model.loadFromOpfs${textOnly ? 'TextOnly' : ''} (size=${size}, max_ctx=${maxContext || 'default'})`);
    model = textOnly
        ? await Model.loadFromOpfsTextOnly(readFn, size, maxContext | 0)
        : await Model.loadFromOpfs(readFn, size);
    log(`load: ready (vocabSize=${model.vocabSize})`);
    // Return the readonly getters in one shot so the main-thread WorkerClient
    // can cache them without round-trips on every access.
    return {
        vocabSize:         model.vocabSize,
        hasVision:         model.hasVision,
        hasAudio:          model.hasAudio,
        imageSentinelIds:  model.imageSentinelIds() ?? null,
        audioSentinelIds:  model.audioSentinelIds() ?? null,
    };
}

function requireModel() {
    if (!model) throw new Error('no model loaded — call load() first');
    return model;
}

const RPC = {
    load:            handleLoad,

    encode:          ({ text })      => Array.from(requireModel().encode(text)),
    step:            async ({ tokenId }) => await requireModel().step(tokenId),
    stepWithEmb:     async ({ embedding }) =>
        await requireModel().stepWithEmbedding(embedding),
    /// One-RPC variant for the generation hot loop: step + isEos + tokenStr
    /// in a single round-trip. Saves 2 postMessage RTTs per token vs calling
    /// the three RPCs separately — adds up at ~120 ms/tok where each RTT
    /// is a couple of milliseconds.
    stepAndDecode:   async ({ tokenId }) => {
        const m = requireModel();
        const next = await m.step(tokenId);
        return {
            next,
            isEos: m.isEos(next),
            str:   m.tokenStr(next) ?? null,
        };
    },
    tokenStr:        ({ id })        => requireModel().tokenStr(id) ?? null,
    isEos:           ({ id })        => requireModel().isEos(id),
    reset:           ()              => requireModel().reset(),
    setSampling:     ({ opts })      => requireModel().setSampling(opts),
    renderChat:      ({ messages, withBos }) =>
        requireModel().renderChat(messages, !!withBos),
    imageSentinelIds:()              => requireModel().imageSentinelIds() ?? null,
    audioSentinelIds:()              => requireModel().audioSentinelIds() ?? null,
    imageSoftTokenCount: ({ h, w })  => requireModel().imageSoftTokenCount(h, w),
    decodeWav:       ({ bytes })     => requireModel().decodeWav(bytes),
    encodeImage:     async ({ pixels, h, w }) =>
        await requireModel().encodeImage(pixels, h, w),
    encodeAudio:     async ({ pcm }) => await requireModel().encodeAudio(pcm),

    free: () => {
        if (model) { try { model.free?.(); } catch (_) {} model = null; }
        if (syncHandle) {
            try { syncHandle.close(); } catch (_) {}
            syncHandle = null;
        }
    },
};

// Per-RPC tracing costs a /api/log beacon per call which adds up across
// step/isEos/tokenStr at every token. Off by default; flip on by setting
// `RULLAMA_RPC_TRACE = '1'` on the worker scope (e.g. via a query string
// the page reads, or just edit here when debugging).
const RPC_TRACE = false;

self.addEventListener('message', async (ev) => {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object' || !msg.type) return;
    const { requestId, type } = msg;
    const handler = RPC[type];
    if (!handler) {
        self.postMessage({ requestId, ok: false, error: `unknown RPC type: ${type}` });
        return;
    }
    if (RPC_TRACE) log(`rpc-start ${type}`);
    try {
        const result = await handler(msg);
        if (RPC_TRACE) log(`rpc-done  ${type}`);
        self.postMessage({ requestId, ok: true, result });
    } catch (e) {
        const err = e?.message ?? String(e);
        // Errors always beacon — these are real signal.
        log(`rpc ${type} failed: ${err}`);
        self.postMessage({ requestId, ok: false, error: err });
    }
});
