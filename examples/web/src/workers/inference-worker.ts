// rullama inference SharedWorker — *router*.
//
// One SharedWorker per origin. Holds the port table (one MessagePort per
// tab), arbitrates the inference session FIFO across tabs, and fans out
// notifications. All wasm / OPFS-sync-handle work lives in a single
// child Dedicated Worker (`inference-core-worker.ts`) — sync access
// handles are spec-restricted to Dedicated Workers, which is why the
// previous "everything in the SharedWorker" shape blew up on `load`.
//
// Wire protocol (port → router), one message per RPC:
//   { requestId, type, ...args }
//
// Wire protocol (router → port):
//   { requestId, ok: true,  result }                     — RPC reply
//   { requestId, ok: false, error }                      — RPC failure
//   { type: "log",    args }                             — debug fanout
//   { type: "notify", kind, ...payload }                 — cross-tab event

import InferenceCoreWorker from "./inference-core-worker?worker";

// ───────────────────────────────────────────────────────────────────────
// Port table
// ───────────────────────────────────────────────────────────────────────

const PORTS = new Set<MessagePort>();
const PORT_LAST_SEEN = new WeakMap<MessagePort, number>();

interface LoadedModelInfo {
    name: string | null;
    modelKey: string;
    filename: string;
    hasVision: boolean;
    hasAudio: boolean;
    vocabSize: number;
    imageSentinelIds: [number, number] | null;
    audioSentinelIds: [number, number] | null;
}
// Mirror of the core worker's loadedInfo, kept in sync via modelLoaded /
// modelFreed notifications. Lets us hand fresh state to a newly
// connecting tab without a round-trip.
let loadedInfo: LoadedModelInfo | null = null;

// ───────────────────────────────────────────────────────────────────────
// Session arbitration (FIFO across tabs)
// ───────────────────────────────────────────────────────────────────────

interface ActiveSession { sid: number; port: MessagePort; }
let active: ActiveSession | null = null;
let nextSid = 1;

interface Waiter {
    port: MessagePort;
    abortToken: string;
    resolve: (sid: number) => void;
    reject:  (e: Error) => void;
}
const queue: Waiter[] = [];

const STATEFUL_RPCS = new Set([
    "load",
    "free",
    "step",
    "stepWithEmb",
    "stepAndDecode",
    "encodeImage",
    "encodeAudio",
    "reset",
    "setSampling",
]);

function acquireSession(abortToken: string, port: MessagePort): Promise<number> {
    return new Promise<number>((resolve, reject) => {
        if (!active) {
            active = { sid: ++nextSid, port };
            resolve(active.sid);
            return;
        }
        queue.push({ port, abortToken, resolve, reject });
    });
}

function releaseSession(sid: number, port: MessagePort): boolean {
    if (!active || active.sid !== sid || active.port !== port) return false;
    active = null;
    wakeNext();
    return true;
}

function cancelAcquire(abortToken: string): boolean {
    const idx = queue.findIndex((w) => w.abortToken === abortToken);
    if (idx < 0) return false;
    const w = queue.splice(idx, 1)[0];
    w.reject(new Error("aborted"));
    return true;
}

function wakeNext() {
    if (active) return;
    const w = queue.shift();
    if (!w) return;
    active = { sid: ++nextSid, port: w.port };
    w.resolve(active.sid);
}

function checkSession(port: MessagePort, sid: unknown): string | null {
    const n = Number(sid);
    if (!active) return "no active session";
    if (active.sid !== n) return `session mismatch: held=${active.sid} called=${n}`;
    if (active.port !== port) return "session not owned by this port";
    return null;
}

// ───────────────────────────────────────────────────────────────────────
// Child Dedicated Worker (wasm + sync handles live here)
// ───────────────────────────────────────────────────────────────────────

const child: Worker = new InferenceCoreWorker();

interface Pending {
    port: MessagePort;
    originalRequestId: number;
}
const pendingChild = new Map<number, Pending>();
let nextChildReqId = 1;

child.addEventListener("message", (ev: MessageEvent) => {
    const msg = ev.data;
    if (!msg || typeof msg !== "object") return;

    // Notifications / logs → fan out to every connected port and (for
    // notifications) update the router's cached state.
    if (msg.type === "log") {
        for (const port of PORTS) {
            try { port.postMessage(msg); } catch { /* */ }
        }
        return;
    }
    if (msg.type === "notify") {
        if (msg.kind === "modelLoaded") {
            const { name, modelKey, filename, hasVision, hasAudio, vocabSize,
                imageSentinelIds, audioSentinelIds } = msg;
            loadedInfo = {
                name: (name as string | null) ?? null,
                modelKey: String(modelKey),
                filename: String(filename),
                hasVision: !!hasVision,
                hasAudio:  !!hasAudio,
                vocabSize: Number(vocabSize) || 0,
                imageSentinelIds: (imageSentinelIds ?? null) as [number, number] | null,
                audioSentinelIds: (audioSentinelIds ?? null) as [number, number] | null,
            };
        } else if (msg.kind === "modelFreed") {
            loadedInfo = null;
        }
        for (const port of PORTS) {
            try { port.postMessage(msg); } catch { /* */ }
        }
        return;
    }

    // RPC reply → route back to the originating port, restoring its
    // requestId.
    if (typeof msg.requestId === "number") {
        const pending = pendingChild.get(msg.requestId);
        if (!pending) return;
        pendingChild.delete(msg.requestId);
        try {
            pending.port.postMessage({
                requestId: pending.originalRequestId,
                ok: msg.ok,
                ...(msg.ok ? { result: msg.result } : { error: msg.error }),
            });
        } catch { /* port gone */ }
    }
});

function forwardToChild(port: MessagePort, originalRequestId: number, type: string, args: Record<string, unknown>) {
    const childReqId = nextChildReqId++;
    pendingChild.set(childReqId, { port, originalRequestId });
    // Strip `requestId` / `type` keys that came from the tab; we send a
    // freshly-tagged message to the child. Any sid baggage is left in
    // place — the router already validated it.
    const { requestId: _r, type: _t, ...rest } = args as { requestId?: number; type?: string } & Record<string, unknown>;
    void _r; void _t;
    child.postMessage({ requestId: childReqId, type, ...rest });
}

// Drop pending child-reply mappings whose origin port has disconnected.
// The reply will still come back from the child, but we'll have nowhere
// to send it — already handled in the child listener by the `if
// (!pending) return` guard, but we proactively clean up here so the map
// doesn't leak.
function dropPendingForPort(port: MessagePort) {
    for (const [id, p] of pendingChild) {
        if (p.port === port) pendingChild.delete(id);
    }
}

// ───────────────────────────────────────────────────────────────────────
// Port lifecycle
// ───────────────────────────────────────────────────────────────────────

function disconnectPort(port: MessagePort, reason: string) {
    if (!PORTS.has(port)) return;
    PORTS.delete(port);
    PORT_LAST_SEEN.delete(port);
    if (active && active.port === port) {
        const sid = active.sid;
        active = null;
        log(`port disconnect (${reason}) released session ${sid}`);
        wakeNext();
    }
    for (let i = queue.length - 1; i >= 0; i--) {
        if (queue[i].port === port) {
            queue[i].reject(new Error("port disconnected"));
            queue.splice(i, 1);
        }
    }
    dropPendingForPort(port);
    try { port.close(); } catch { /* */ }
}

const HEARTBEAT_GC_INTERVAL = 15_000;
const HEARTBEAT_DEAD_MS      = 30_000;
setInterval(() => {
    const now = Date.now();
    for (const port of PORTS) {
        const seen = PORT_LAST_SEEN.get(port) ?? now;
        if (now - seen > HEARTBEAT_DEAD_MS) {
            disconnectPort(port, "heartbeat timeout");
        }
    }
}, HEARTBEAT_GC_INTERVAL);

// ───────────────────────────────────────────────────────────────────────
// Logging (the router itself uses this too)
// ───────────────────────────────────────────────────────────────────────

function log(...args: unknown[]) {
    const argStrs = args.map((a) => String(a));
    for (const port of PORTS) {
        try { port.postMessage({ type: "log", args: argStrs }); } catch { /* */ }
    }
}

// ───────────────────────────────────────────────────────────────────────
// RPC dispatch
// ───────────────────────────────────────────────────────────────────────

type Args = Record<string, unknown>;

async function handleRequest(
    port: MessagePort,
    raw: { requestId: number; type: string } & Args,
) {
    if (!raw || typeof raw !== "object" || !raw.type) return;
    const { requestId, type } = raw;
    PORT_LAST_SEEN.set(port, Date.now());

    const reply = (payload: Record<string, unknown>) => {
        try { port.postMessage({ requestId, ...payload }); } catch { /* */ }
    };

    try {
        switch (type) {
            // ── Router-only RPCs ────────────────────────────────────────
            case "ping":
                reply({ ok: true, result: true });
                return;
            case "disconnect":
                reply({ ok: true, result: true });
                disconnectPort(port, "client disconnect");
                return;
            case "acquireSession": {
                const sid = await acquireSession(String(raw.abortToken ?? ""), port);
                reply({ ok: true, result: sid });
                return;
            }
            case "releaseSession": {
                const ok = releaseSession(Number(raw.sid), port);
                reply({ ok: true, result: ok });
                return;
            }
            case "cancelAcquire": {
                const ok = cancelAcquire(String(raw.abortToken));
                reply({ ok: true, result: ok });
                return;
            }
            case "currentMeta":
                reply({ ok: true, result: { loaded: loadedInfo, activeSessionPortHeld: active != null } });
                return;
        }

        // ── Stateful RPCs: enforce session ownership before forwarding ─
        if (STATEFUL_RPCS.has(type)) {
            const err = checkSession(port, raw.sid);
            if (err) {
                reply({ ok: false, error: err });
                return;
            }
        }

        // ── Forward to child Dedicated Worker ───────────────────────────
        forwardToChild(port, requestId, type, raw);
    } catch (e) {
        const err = (e as Error)?.message ?? String(e);
        reply({ ok: false, error: err });
    }
}

// ───────────────────────────────────────────────────────────────────────
// SharedWorker entry — one onconnect per tab
// ───────────────────────────────────────────────────────────────────────

(self as unknown as SharedWorkerGlobalScope).onconnect = (e: MessageEvent) => {
    const port = (e.ports as MessagePort[])[0];
    if (!port) return;
    port.start();
    PORTS.add(port);
    PORT_LAST_SEEN.set(port, Date.now());
    port.addEventListener("message", (ev: MessageEvent) => {
        void handleRequest(port, ev.data);
    });
    // Push initial state so the freshly connected tab can skip "Load a
    // model" if a model is already active in the core worker.
    port.postMessage({
        type: "notify",
        kind: "meta",
        loaded: loadedInfo,
    });
};
