import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { getClient } from "@/lib/inference";
import { ensureFreshServiceWorker } from "@/lib/pwa";
import { installGlobalRestartListeners } from "@/lib/restart";
import { ToastProvider } from "@/lib/toast";
import { Toaster } from "@/components/Toaster";

// Automation hook for safaridriver-driven debugging on iPhone.
// Gated by `?automation=1` URL param so the surface is invisible in
// normal use. Exposes the worker client + a canned end-to-end repro
// flow (download → load → train) so a shell script can crash-debug
// without driving fragile React DOM selectors.
if (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("automation") === "1") {
    interface ReproState {
        phase:    string;
        downloadBytes?:  number;
        downloadTotal?:  number;
        sessionStarted?: boolean;
        loadedAt?:       number;
        trainingStartedAt?: number;
        trainingStep?:   number;
        trainingError?:  string | null;
        finishedAt?:     number;
    }
    interface RullamaAutomation {
        client:     ReturnType<typeof getClient>;
        ready:      Promise<void>;
        dumpLogs:   (id?: string) => Promise<string>;
        latestId:   () => Promise<string | null>;
        crashedId:  () => Promise<string | null>;
        runRepro:   () => Promise<void>;
        reproState: ReproState;
    }
    const w = window as unknown as { __rullama?: RullamaAutomation };
    const state: ReproState = { phase: "idle" };
    const client = getClient();
    w.__rullama = {
        client,
        reproState: state,
        ready:     new Promise<void>((resolve) => setTimeout(resolve, 500)),
        latestId:   async () => {
            const list = await client.logs.list();
            return list[0]?.id ?? null;
        },
        crashedId:  async () => {
            const list = await client.logs.list();
            const cur  = await client.logs.currentId().catch(() => "");
            return list.find((s) => s.id !== cur && !s.cleanExit)?.id ?? null;
        },
        dumpLogs:   async (id?: string) => {
            const target = id ?? (await client.logs.list())[0]?.id;
            if (!target) return "(no sessions)";
            return await client.logs.read(target);
        },
        runRepro:   async () => {
            try {
                // 1. Discover model.
                state.phase = "discovering";
                const models = await fetch("/api/models").then((r) => r.json()) as Array<{
                    name: string; modelKey: string; filename: string; size: number;
                }>;
                const m = models.find((x) => x.name === "gemma4:e2b") ?? models[0];
                if (!m) throw new Error("no models discovered via /api/models");

                // 2. Wire download progress notifications BEFORE firing
                //    ensureModel so we don't miss the early bytes.
                client.subscribe("downloadProgress", (p) => {
                    if (p.modelKey === m.modelKey && p.filename === m.filename) {
                        state.downloadBytes = Number(p.bytesWritten);
                        state.downloadTotal = Number(p.totalBytes);
                    }
                });

                // 3. Hold a session for the entire download + load +
                //    training lifecycle. Training start consumes the
                //    Model; we never need to release until done.
                state.phase = "acquiring_session";
                await client.acquireSession();
                state.sessionStarted = true;

                // 4. Download (~7 GB). Prefer the Cloudflare R2 CDN URL
                //    (fast, off the iPhone's own WiFi internet) over the
                //    Mac-served /api/blob (slow Wireless-N LAN leg). R2
                //    CORS must allow this origin — see api.ts R2_HOST.
                state.phase = "downloading";
                state.downloadTotal = m.size;
                const r2Url = m.name === "gemma4:e2b"
                    ? "https://models.brainwires.dev/gemma4-e2b.gguf"
                    : `/api/blob/${encodeURIComponent(m.name)}`;
                await client.ensureModel({
                    url:          r2Url,
                    modelKey:     m.modelKey,
                    filename:     m.filename,
                    expectedSize: m.size,
                });

                // 5. Load text-only (no vision/audio towers). max_context=512
                //    matches the validated iPhone training path.
                state.phase = "loading";
                await client.load(m.modelKey, m.filename, {
                    name:       m.name,
                    maxContext: 512,
                    textOnly:   true,
                });
                state.loadedAt = Date.now();

                // 6. Tokenize a small training example. One example is
                //    enough to trigger the crash signature we're hunting.
                state.phase = "tokenizing";
                const ids = await client.encode("Hello, world. This is a test.");
                if (ids.length < 4) throw new Error(`encode produced too few tokens (${ids.length})`);

                // 7. Start training with the Memory-tight (iPhone-safe)
                //    preset — same JSON shape the FineTunePanel emits.
                state.phase = "training_start";
                await client.trainingStart({
                    loraConfig: {
                        rank:           1,
                        alpha:          2,
                        dropout:        0,
                        target_modules: ["attn_q", "attn_v"],
                    } as unknown as Parameters<typeof client.trainingStart>[0]["loraConfig"],
                    hparams: {
                        epochs:                       1,
                        batch_size:                   1,
                        warmup_steps:                 0,
                        weight_decay:                 0,
                        lr_scheduler:                 "constant",
                        seed:                         12648430,
                        gradient_accumulation_steps:  1,
                        mixed_precision:              false,
                        backward_layer_floor:         25,
                        learning_rate:                0.0003,
                        max_seq_len:                  32,
                        max_grad_norm:                1,
                        loss_mode:                    "per_position",
                        gradient_checkpointing:       true,
                    } as unknown as Parameters<typeof client.trainingStart>[0]["hparams"],
                    totalSteps: 1,
                });
                state.trainingStartedAt = Date.now();

                // 8. One step in per_position mode, target = next-token
                //    shift of inputIds (simplest valid target tensor).
                state.phase = "training_step";
                state.trainingStep = 1;
                const targets = new Uint32Array(ids.length);
                for (let i = 0; i < ids.length - 1; i++) targets[i] = ids[i + 1];
                targets[ids.length - 1] = ids[ids.length - 1];
                const result = await client.trainingStep({
                    inputIds: ids,
                    targets,
                    lossMode: "per_position",
                });
                state.phase = "succeeded";
                state.finishedAt = Date.now();
                console.log("[automation] training step succeeded:", result);
            } catch (e) {
                state.phase = "errored";
                state.trainingError = (e as Error)?.message ?? String(e);
                state.finishedAt = Date.now();
                console.error("[automation] repro errored:", e);
            }
        },
    };
}

// Install before anything else mounts so that any dynamic-import or
// chunk-load failure (Vite-emitted asset URL that doesn't match the
// active service worker's precache after a deploy) routes through the
// restart overlay rather than landing as a silent console error.
installGlobalRestartListeners();

// Gate React render on service-worker freshness. Returning visitors with
// a pending update wait (≤1.5 s) for the new SW to claim the page so the
// first render is already against the fresh asset set. First-ever loads
// fall through immediately. See `lib/pwa.ts` for the lifecycle details.
//
// Wrapped in try/catch as a safety net: a thrown error here would black-
// screen the PWA (top-level await rejects → bundle never resolves →
// React never mounts → static-HTML watchdog catches it after 8 s). We'd
// rather render in degraded mode against possibly-stale assets than
// trip the watchdog on a transient SW lifecycle hiccup.
try {
    await ensureFreshServiceWorker();
} catch (e) {
    // Surface to the console for diagnosis but DO NOT abort boot.
    // eslint-disable-next-line no-console
    console.warn("[rullama] ensureFreshServiceWorker threw:", e);
}

// (Previously this called installPostBootSwReloadListener to reload
// on controllerchange. That was a workaround for autoUpdate-mode +
// precache-nav. With prompt-mode + NetworkFirst nav + the version-
// manifest update flow, there's nothing to do here — the user clicks
// the banner and a coordinated reload happens explicitly.)

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <ToastProvider>
            <App />
            <Toaster />
        </ToastProvider>
    </React.StrictMode>,
);
