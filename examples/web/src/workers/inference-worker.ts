// rullama inference SharedWorker — *router* with host election.
//
// One SharedWorker per origin. Holds the port table (one MessagePort
// per tab), arbitrates the inference session FIFO across tabs, and
// fans out notifications. The wasm Model, OPFS sync handles, and chat
// DB live in a *single* child Dedicated Worker — but SharedWorker
// can't spawn workers in any browser, so the **host tab's main
// thread** spawns the core Dedicated Worker for us and hands its
// MessagePort to this router via `attachCore`. If the host tab
// closes, the router picks the next connected tab and asks it to
// spawn a fresh core (which then re-loads the model).
//
// Wire protocol (port → router), one message per RPC:
//   { requestId, type, ...args }
//
// Wire protocol (router → port):
//   { requestId, ok: true,  result }                     — RPC reply
//   { requestId, ok: false, error }                      — RPC failure
//   { type: "log",    args }                             — debug fanout
//   { type: "notify", kind, ...payload }                 — cross-tab event
//
// Special notify kinds:
//   pleaseSpawnCore — sent to one tab when no core is attached
//   coreReady       — fanned out once the router has a fresh corePort

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
// Mirror of the core worker's loadedInfo, updated via modelLoaded /
// modelFreed notifications. Lets us hand fresh state to a newly
// connecting tab without a round-trip.
let loadedInfo: LoadedModelInfo | null = null;

// Cross-tab update coordination. One tab's boot-time version check
// detects an update and broadcasts `updateAvailable`; the router
// remembers the version so newly-connecting tabs get the same banner
// without each having to re-fetch /version.json. When any tab clicks
// "Apply now", the router broadcasts `applyingUpdate` to ALL ports
// and signals the dedicated core worker to shut down. `updateInProgress`
// gates further actions (e.g. don't elect a new host mid-shutdown).
let pendingUpdateVersion: string | null = null;
let updateInProgress = false;

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
    "releaseVisionWeights",
    "releaseAudioWeights",
    "step",
    "stepWithEmb",
    "stepAndDecode",
    "encodeImage",
    "encodeAudio",
    "transcribeAudio",
    "reset",
    "setSampling",
    "saveKvState",
    "restoreKvState",
    // Training RPCs all mutate the Model handle (TrainingSession owns
    // it for the session's lifetime) — same session-locking pattern as
    // the chat-side step/encode RPCs.
    "trainingProbeFit",
    "trainingStart",
    "trainingStep",
    "trainingZeroGrads",
    "trainingForwardBackward",
    "trainingOptimizerStep",
    "trainingSaveAdapter",
    "trainingSaveAdapterAndFinish",
    "trainingCancel",
    "trainingFinish",
    "trainingApplyAdapter",
    "trainingClearAdapter",
    "trainingDeleteAdapter",
]);
// renderChatForContinuation + position + trainingStatus + trainingListAdapters
// are stateless reads — no session gating needed; they can interleave with
// another tab's generation.

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
// Core election
// ───────────────────────────────────────────────────────────────────────

let corePort: MessagePort | null = null;
let coreHostPort: MessagePort | null = null;
let electionInFlight: { tab: MessagePort; startedAt: number } | null = null;

const ELECTION_TIMEOUT_MS = 15_000;

interface PendingChild {
    port: MessagePort;
    originalRequestId: number;
}
const pendingChild = new Map<number, PendingChild>();
let nextChildReqId = 1;

// Liveness pings (router → core). Disjoint id space from pendingChild
// so onCoreMessage can route the pong reply without confusing it with
// a regular tab RPC reply. See `verifyCoreLive`.
const pingsAwaitingPong = new Map<number, (ok: boolean) => void>();
let nextPingId = 1_000_000_000;
const CORE_PING_TIMEOUT_MS = 800;

function electHost(tab: MessagePort) {
    if (corePort || electionInFlight) return;
    electionInFlight = { tab, startedAt: Date.now() };
    try { tab.postMessage({ type: "notify", kind: "pleaseSpawnCore" }); } catch { /* */ }
}

function attachCore(hostPort: MessagePort, transferredPort: MessagePort) {
    if (corePort) {
        // Already attached; ignore stray attachCore from a tab that
        // didn't get the memo yet.
        try { transferredPort.close(); } catch { /* */ }
        return;
    }
    corePort = transferredPort;
    coreHostPort = hostPort;
    electionInFlight = null;
    corePort.addEventListener("message", onCoreMessage);
    corePort.start();
    log(`core: host attached via tab port`);
    notifyAll({ type: "notify", kind: "coreReady" });
}

function onCoreMessage(ev: MessageEvent) {
    const msg = ev.data;
    if (!msg || typeof msg !== "object") return;

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

    // Pong for an internal liveness ping — routed back to the waiting
    // verifyCoreLive() Promise instead of any tab.
    if (typeof msg.requestId === "number" && pingsAwaitingPong.has(msg.requestId)) {
        const resolver = pingsAwaitingPong.get(msg.requestId)!;
        pingsAwaitingPong.delete(msg.requestId);
        resolver(!!msg.ok);
        return;
    }

    // RPC reply
    if (typeof msg.requestId === "number") {
        const p = pendingChild.get(msg.requestId);
        if (!p) return;
        pendingChild.delete(msg.requestId);
        try {
            p.port.postMessage({
                requestId: p.originalRequestId,
                ok: msg.ok,
                ...(msg.ok ? { result: msg.result } : { error: msg.error }),
            });
        } catch { /* */ }
    }
}

/**
 * Send a `pingCore` over the current corePort and wait up to
 * `CORE_PING_TIMEOUT_MS` for the reply. Used on new tab connections to
 * detect the "host tab died without firing pagehide" case (common on
 * iOS Safari when the WebContent process is killed) — without this,
 * the router keeps trusting a corePort whose other end is dead until
 * the 30s heartbeat GC runs, and RPCs forwarded in that window vanish
 * into the closed port. Returns true on pong, false on timeout / no
 * corePort / postMessage throw.
 */
function verifyCoreLive(): Promise<boolean> {
    if (!corePort) return Promise.resolve(false);
    const port = corePort;
    const pingId = nextPingId++;
    return new Promise<boolean>((resolve) => {
        let done = false;
        const finish = (ok: boolean) => {
            if (done) return;
            done = true;
            pingsAwaitingPong.delete(pingId);
            resolve(ok);
        };
        pingsAwaitingPong.set(pingId, finish);
        setTimeout(() => finish(false), CORE_PING_TIMEOUT_MS);
        try { port.postMessage({ requestId: pingId, type: "pingCore" }); }
        catch { finish(false); }
    });
}

function loseCore(reason: string) {
    if (!corePort && !coreHostPort) return;
    log(`core: lost (${reason})`);
    try { corePort?.close(); } catch { /* */ }
    corePort = null;
    coreHostPort = null;
    electionInFlight = null;
    // Reject all in-flight child requests; clients retry-once after coreReady.
    for (const [, p] of pendingChild) {
        try {
            p.port.postMessage({
                requestId: p.originalRequestId,
                ok: false,
                error: "core disconnected",
            });
        } catch { /* */ }
    }
    pendingChild.clear();
    if (loadedInfo) {
        loadedInfo = null;
        notifyAll({ type: "notify", kind: "modelFreed" });
    }
    // Pick a successor.
    const next = nextElectableTab();
    if (next) electHost(next);
}

function nextElectableTab(): MessagePort | null {
    for (const p of PORTS) return p;
    return null;
}

function forwardToCore(port: MessagePort, originalRequestId: number, type: string, args: Record<string, unknown>) {
    if (!corePort) {
        // No core attached yet — reply with a transient error. The
        // client retries on coreReady.
        try {
            port.postMessage({
                requestId: originalRequestId,
                ok: false,
                error: "core disconnected",
            });
        } catch { /* */ }
        return;
    }
    const childReqId = nextChildReqId++;
    pendingChild.set(childReqId, { port, originalRequestId });
    const { requestId: _r, type: _t, ...rest } =
        args as { requestId?: number; type?: string } & Record<string, unknown>;
    void _r; void _t;
    try {
        corePort.postMessage({ requestId: childReqId, type, ...rest });
    } catch (e) {
        pendingChild.delete(childReqId);
        loseCore(`postMessage failed: ${(e as Error).message}`);
    }
}

function dropPendingForPort(port: MessagePort) {
    for (const [id, p] of pendingChild) {
        if (p.port === port) pendingChild.delete(id);
    }
}

// ───────────────────────────────────────────────────────────────────────
// Port lifecycle
// ───────────────────────────────────────────────────────────────────────

function notifyAll(msg: Record<string, unknown>) {
    for (const port of PORTS) {
        try { port.postMessage(msg); } catch { /* */ }
    }
}

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

    // If this was the core host, we lose the core too.
    if (coreHostPort === port) {
        loseCore("host tab disconnected");
    }
    // If this tab was mid-election, restart election with whatever's left.
    if (electionInFlight && electionInFlight.tab === port) {
        electionInFlight = null;
        const next = nextElectableTab();
        if (next) electHost(next);
    }
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
    // Election timeout: if the elected tab hasn't called attachCore
    // within the window, move on. (Mobile Safari can throttle tabs.)
    if (electionInFlight
        && Date.now() - electionInFlight.startedAt > ELECTION_TIMEOUT_MS) {
        const stuck = electionInFlight.tab;
        log(`election: tab did not attachCore within ${ELECTION_TIMEOUT_MS}ms`);
        electionInFlight = null;
        const next = nextElectableTab();
        if (next && next !== stuck) electHost(next);
        else if (next) electHost(next); // sole tab; try again anyway
    }
}, HEARTBEAT_GC_INTERVAL);

// ───────────────────────────────────────────────────────────────────────
// Logging
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
    rawEvent?: MessageEvent,
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
                reply({ ok: true, result: { loaded: loadedInfo, activeSessionPortHeld: active != null, pendingUpdateVersion } });
                return;
            case "updateAvailable": {
                // A tab detected (via its boot-time version-manifest
                // fetch) that a newer build is deployed. Remember it,
                // and broadcast so every other open tab can surface
                // the banner too — no need for each tab to re-fetch.
                const v = String(raw.version ?? "");
                if (!v) {
                    reply({ ok: false, error: "updateAvailable: missing version" });
                    return;
                }
                pendingUpdateVersion = v;
                notifyAll({ type: "notify", kind: "updateAvailable", version: v });
                reply({ ok: true, result: true });
                return;
            }
            case "applyUpdate": {
                // Coordinated multi-tab shutdown + reload. Every tab
                // shows the ApplyingOverlay and reloads in lockstep, so
                // the new SharedWorker URL (Vite-hashed) is picked up
                // simultaneously and we never end up with a v1/v2 split.
                if (updateInProgress) {
                    reply({ ok: true, result: false });
                    return;
                }
                updateInProgress = true;
                const v = pendingUpdateVersion ?? String(raw.version ?? "");
                notifyAll({ type: "notify", kind: "applyingUpdate", version: v });
                // Tell the dedicated core worker to release OPFS / GPU /
                // DB / Model BEFORE the tabs reload. The core's shutdown
                // handler runs `releaseAllHandles()` and then
                // `self.close()`s — same path used on pagehide. If
                // corePort is null or already closed (host tab died
                // before reaching here), tabs still reload via the
                // applyingUpdate broadcast above and a fresh core is
                // elected on the new bundle.
                if (corePort) {
                    try { corePort.postMessage({ type: "shutdown" }); }
                    catch (e) { log(`applyUpdate: core shutdown post failed (port closed?): ${(e as Error).message ?? e}`); }
                }
                reply({ ok: true, result: true });
                return;
            }
            case "attachCore": {
                // Transferred MessagePort arrives on the event.ports array.
                const transferred = rawEvent && rawEvent.ports && rawEvent.ports[0];
                if (!transferred) {
                    reply({ ok: false, error: "attachCore: no transferred port" });
                    return;
                }
                attachCore(port, transferred);
                reply({ ok: true, result: true });
                return;
            }
        }

        if (STATEFUL_RPCS.has(type)) {
            const err = checkSession(port, raw.sid);
            if (err) {
                reply({ ok: false, error: err });
                return;
            }
        }

        forwardToCore(port, requestId, type, raw);
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
        void handleRequest(port, ev.data, ev);
    });
    // Push initial state so the freshly connected tab can skip "Load a
    // model" if a model is already active in the core worker. We also
    // include any pendingUpdateVersion that another tab already
    // detected, so this tab can surface its banner without an
    // independent /version.json fetch (the banner is still gated on
    // busy=false in the React layer).
    port.postMessage({
        type: "notify",
        kind: "meta",
        loaded: loadedInfo,
        pendingUpdateVersion,
    });
    // If we already have a core, verify it's actually alive before
    // telling the new tab so. The "host tab killed without firing
    // pagehide" case (iOS WebContent process death, refresh during
    // backgrounding, page-reaper) leaves a corePort whose other end is
    // closed; the router has no socket-level way to detect that and
    // would otherwise trust it until the 30s heartbeat GC runs. A
    // quick pingCore round-trip catches it in <1s — on timeout we
    // synthesise a loseCore so the new tab inherits a clean election
    // path instead of having its first RPC vanish into the closed port.
    if (corePort) {
        void (async () => {
            const alive = await verifyCoreLive();
            if (alive) {
                try { port.postMessage({ type: "notify", kind: "coreReady" }); }
                catch { /* */ }
                return;
            }
            log(`core: ping failed on new connection — re-electing`);
            loseCore("ping failed on new connection");
            // loseCore already calls electHost on the next electable
            // tab. If, somehow, no election fired (e.g. PORTS was empty
            // at loseCore time, which can't happen here since we just
            // added `port`), kick one off manually.
            if (!corePort && !electionInFlight) electHost(port);
        })();
    } else if (!electionInFlight) {
        // No core attached and no election running → this tab gets to
        // spawn the core.
        electHost(port);
    }
};
