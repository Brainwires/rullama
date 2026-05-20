// Fine-tune tab — drives in-browser LoRA training over the loaded
// Model. Mirrors the CLI dials in `crates/rullama-finetune/examples/
// train_jsonl.rs` so users moving between the two recognise the
// surface. Designed states: no model / no dataset / ready / training /
// complete / error — each with its own affordance, never a fallback.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
    Play, Square, Save, CheckCircle2, AlertTriangle, Sparkles,
    FileText, Settings2, Activity, RefreshCw, Trash2, Plug,
} from "lucide-react";
import { cn, clampInt, clampNum } from "@/lib/utils";
import {
    getClient,
    type TrainingLoraConfig, type TrainingHyperparams,
    type TrainingStepReport, type AdapterListEntry,
} from "@/lib/inference";
import { useToast } from "@/lib/toast";
import type { ModelStatus } from "@/components/ModelLoader";
import { TrainingProgress, type TrainingProgressState } from "@/components/TrainingProgress";

// Thin wrappers around the toast context so the rest of the file reads
// like `toast.success(...)` without leaking the `showToast({level,...})`
// shape everywhere.
function useFineTuneToasts() {
    const { showToast } = useToast();
    return useMemo(() => ({
        success: (msg: string) => { showToast({ level: "success", title: msg }); },
        info:    (msg: string) => { showToast({ level: "info",    title: msg }); },
        warn:    (msg: string) => { showToast({ level: "warn",    title: msg }); },
        error:   (msg: string) => { showToast({ level: "error",   title: msg }); },
    }), [showToast]);
}

interface Props {
    modelStatus: ModelStatus;
    activeAdapter: string | null;
    onAdapterChanged?: (name: string | null) => void;
}

interface ParsedExample { prompt: string; completion: string }

interface RecentStep extends TrainingStepReport { ms: number }

type Phase = "idle" | "ready" | "training" | "stopping" | "done" | "error";

const DEFAULT_TARGETS = ["attn_q", "attn_k", "attn_v", "attn_o"];
const ALL_TARGETS = ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down"];

// Device-aware defaults computed once at mount. Phones get a tighter
// rank + shorter seq + gradient checkpointing on by default; desktops
// get the headroom for richer adapters.
//
// Default `max_seq_len` is intentionally conservative. The trainer
// allocates per-layer activation captures sized as seq * per_position,
// summed across 17 buffers per layer × n_layers — for gemma4-e2b
// that's roughly `seq * 40 MB` of GPU memory total. Desktop 128 → ~5 GB
// budget hit, mobile 32 → ~1.3 GB. These are safe starting points;
// users with headroom can crank seq via the form.
function deviceDefaults(): { hp: TrainingHyperparams; lora: TrainingLoraConfig; tight: boolean } {
    const nav = navigator as Navigator & { deviceMemory?: number };
    const memGB = nav.deviceMemory ?? 8;
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const tight = isMobile || memGB < 4;
    return {
        lora: {
            rank: tight ? 4 : 8,
            alpha: tight ? 8 : 16,
            dropout: 0,
            target_modules: [...DEFAULT_TARGETS],
        },
        hp: {
            epochs: 1,
            batch_size: 1,
            learning_rate: 1e-3,
            warmup_steps: 0,
            weight_decay: 0,
            lr_scheduler: "constant",
            seed: 0xC0FFEE,
            max_seq_len: tight ? 32 : 128,
            gradient_accumulation_steps: 1,
            max_grad_norm: 0,
            loss_mode: "next_token",
            // Gradient checkpointing on by default everywhere — the
            // shared-scratch refactor proved bit-identical gradients
            // vs the standard path (1.1M elements, max_diff=0.000e0)
            // while collapsing per-layer activation captures from
            // ~10 MB × n_layers to one shared set. Trades one extra
            // forward replay per layer's backward for ~10× memory
            // savings; the right call on every device.
            gradient_checkpointing: true,
            mixed_precision: false,
            // 0 = backprop every layer (full standard training).
            // The Memory-tight preset overrides this with 25 (last
            // ~10 layers only) when the user opts in.
            backward_layer_floor: 0,
        },
        tight,
    };
}

/** Smallest config that's expected to fit on iPhone 16e (A18 / ~3-4 GB
 *  WebContent budget) without OOM. Rank 1, attn_q + attn_v only,
 *  seq_len 16, gradient checkpointing on. Memory feasibility audit
 *  estimates peak ~2.1 GB on top of the text tower. Applied when the
 *  user toggles the "Memory-tight" switch; the sliders lock so the
 *  preset can't be drifted out of by accident. */
const ULTRA_SAFE_LORA: Pick<TrainingLoraConfig, "rank" | "alpha" | "target_modules" | "dropout"> = {
    rank: 1,
    alpha: 2,
    target_modules: ["attn_q", "attn_v"],
    dropout: 0,
};
const ULTRA_SAFE_HP: Pick<TrainingHyperparams, "max_seq_len" | "batch_size" | "loss_mode" | "gradient_checkpointing" | "backward_layer_floor"> = {
    max_seq_len: 16,
    batch_size: 1,
    loss_mode: "next_token",
    gradient_checkpointing: true,
    // Truncated backward: train only the top 10 layers on gemma4:e2b
    // (35 total → floor=25). The Rust side saturate-clamps if the
    // model has fewer layers, so this is safe across model sizes.
    backward_layer_floor: 25,
};

function parseJsonl(text: string): { examples: ParsedExample[]; errors: string[] } {
    const examples: ParsedExample[] = [];
    const errors: string[] = [];
    text.split(/\r?\n/).forEach((rawLine, i) => {
        const line = rawLine.trim();
        if (!line) return;
        try {
            const obj = JSON.parse(line);
            if (typeof obj.prompt === "string" && typeof obj.completion === "string") {
                examples.push({ prompt: obj.prompt, completion: obj.completion });
            } else if (typeof obj.instruction === "string") {
                const prompt = obj.input
                    ? `${obj.instruction}\n${obj.input}`
                    : obj.instruction;
                examples.push({ prompt, completion: String(obj.output ?? "") });
            } else if (Array.isArray(obj.messages)) {
                // Trivial messages format: take last assistant as completion,
                // concatenate everything else as prompt.
                const msgs = obj.messages as Array<{ role: string; content: string }>;
                const last = msgs[msgs.length - 1];
                if (last?.role === "assistant") {
                    const prompt = msgs.slice(0, -1).map((m) => `${m.role}: ${m.content}`).join("\n");
                    examples.push({ prompt, completion: last.content });
                } else {
                    errors.push(`Line ${i + 1}: messages array must end with an assistant turn`);
                }
            } else {
                errors.push(`Line ${i + 1}: needs prompt+completion, instruction+output, or messages`);
            }
        } catch (e) {
            errors.push(`Line ${i + 1}: ${(e as Error).message}`);
        }
    });
    return { examples, errors };
}

export function FineTunePanel({ modelStatus, activeAdapter, onAdapterChanged }: Props) {
    const toast = useFineTuneToasts();
    const client = useMemo(() => getClient(), []);

    // Dataset + tokenisation.
    const [datasetName, setDatasetName] = useState<string | null>(null);
    const [examples, setExamples] = useState<ParsedExample[]>([]);
    const [parseErrors, setParseErrors] = useState<string[]>([]);
    const [tokenLengths, setTokenLengths] = useState<number[] | null>(null);

    // Form state — initialised once.
    const initialDefaults = useMemo(deviceDefaults, []);
    const [lora, setLora] = useState<TrainingLoraConfig>(initialDefaults.lora);
    const [hp, setHp] = useState<TrainingHyperparams>(initialDefaults.hp);
    const [stepsBudget, setStepsBudget] = useState<number>(100);
    const [adapterName, setAdapterName] = useState<string>("");

    // **B2 — Memory-tight preset.** When on, force-applies the
    // ULTRA_SAFE_LORA + ULTRA_SAFE_HP values and locks the
    // affected sliders. Auto-on for devices `deviceDefaults` flagged
    // as tight (mobile / <4 GB system RAM); user can override either
    // direction. Stored as a usePersistedState-style flag would be
    // nice for follow-up but isn't load-bearing — boot defaults are
    // device-derived, the user's choice survives within the session.
    const [memoryTight, setMemoryTight] = useState<boolean>(initialDefaults.tight);
    const applyMemoryTight = useCallback((on: boolean) => {
        setMemoryTight(on);
        if (on) {
            setLora((cur) => ({ ...cur, ...ULTRA_SAFE_LORA, target_modules: [...ULTRA_SAFE_LORA.target_modules] }));
            setHp((cur) => ({ ...cur, ...ULTRA_SAFE_HP }));
        }
        // Toggling OFF leaves the current values in place — user can
        // then adjust the sliders freely. Auto-reverting to the
        // device default would surprise them.
    }, []);
    // Apply the preset on initial mount when the device flagged tight,
    // so the first paint shows the safe config (not the unsafe
    // deviceDefaults that haven't been overridden yet).
    useEffect(() => {
        if (initialDefaults.tight) {
            setLora((cur) => ({ ...cur, ...ULTRA_SAFE_LORA, target_modules: [...ULTRA_SAFE_LORA.target_modules] }));
            setHp((cur) => ({ ...cur, ...ULTRA_SAFE_HP }));
        }
        // Run once on mount — deliberately no deps.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Runtime state.
    const [phase, setPhase] = useState<Phase>("idle");
    const [recent, setRecent] = useState<RecentStep[]>([]);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const cancelRef = useRef(false);

    // Live mid-step progress fed by the worker's `trainingProgress`
    // notify. Mirrors how Chat renders PipelineProgress for vision
    // encode. `coldHint` flips on after 5 s in the `starting` phase
    // to explain the WGSL-compile pause on first step.
    const [progress, setProgress] = useState<TrainingProgressState | null>(null);
    const [coldHint, setColdHint] = useState<string | null>(null);
    useEffect(() => {
        const offs = [
            client.subscribe("trainingProgress", (p) => {
                const phase = String(p.phase ?? "");
                // Once any progress beacon lands, the cold-start window
                // is over — clear the hint.
                setColdHint(null);
                setProgress({
                    phase: phase as TrainingProgressState["phase"],
                    current: Number(p.current ?? 0),
                    total: Number(p.total ?? 0),
                    step: Number(p.step ?? 0) || undefined,
                    lr: typeof p.lr === "number" ? p.lr : undefined,
                });
            }),
            // **A4 — surface GPU faults during training.** Chat has
            // its own `gpuFault` subscriber in App.tsx; without one
            // here, an OOM / device-lost during `trainingStep` shows
            // up as a generic "Training stopped" toast and the user
            // has no idea it was hardware-related. Subscribing means
            // we get the typed kind + during fields and can produce
            // an actionable banner. Only react when the fault was
            // raised during a training-prefixed RPC so chat-side
            // faults don't bleed into the FineTune panel.
            client.subscribe("gpuFault", (p) => {
                const during = String((p as { during?: unknown }).during ?? "");
                if (!during.startsWith("training")) return;
                const kind = String((p as { kind?: unknown }).kind ?? "unknown");
                const message = String((p as { message?: unknown }).message ?? "");
                const title = kind === "oom"
                    ? "GPU ran out of memory"
                    : kind === "device-lost"
                        ? "GPU device was lost"
                        : "GPU error during training";
                const hint = kind === "oom"
                    ? "Lower the rank / seq_len / target modules and try again. The 'Memory-tight' preset is the smallest safe config on iPhone."
                    : kind === "device-lost"
                        ? "Reload the page to restart the WebGPU context, then try a smaller config."
                        : "Reload and check the dev console for details.";
                toast.error(title);
                setErrorMsg(`${title}. ${hint} (GPU said: ${message})`);
                setPhase("error");
                setProgress(null);
                // Release the Model back to chat so the user isn't
                // stuck. trainingFinish is idempotent + tolerates the
                // session being in a bad state.
                client.trainingFinish().catch((e) => {
                    console.warn("[fine-tune] trainingFinish after gpuFault rejected:", e);
                });
            }),
        ];
        return () => { for (const off of offs) off(); };
    }, [client, toast]);

    // Adapter library.
    const [adapters, setAdapters] = useState<AdapterListEntry[]>([]);
    const refreshAdapters = useCallback(async () => {
        try {
            const r = await client.trainingListAdapters();
            setAdapters(r.entries);
            onAdapterChanged?.(r.active);
        } catch { /* */ }
    }, [client, onAdapterChanged]);
    useEffect(() => { void refreshAdapters(); }, [refreshAdapters]);

    // Adapter parameter-count estimate (so user sees the consequence of
    // toggling FFN targets live, before training).
    const estimatedAdapterMB = useMemo(() => {
        // Rough estimate: rank * 2 * (avg in+out dim) * f32 bytes * n_layers
        // For e2b/e4b: d_model 1536, n_heads*head_dim 2048, n_kv*head_dim 512,
        // ffn 6912, 35 layers. Adam multiplies state by 3x (param + m + v).
        const perModule = (inDim: number, outDim: number) =>
            lora.rank * (inDim + outDim) * 4;
        let bytes = 0;
        const n = 35;
        for (const m of lora.target_modules) {
            const v = m === "attn_q" ? perModule(1536, 2048)
                : m === "attn_k" || m === "attn_v" ? perModule(1536, 512)
                : m === "attn_o" ? perModule(2048, 1536)
                : m === "ffn_gate" || m === "ffn_up" ? perModule(1536, 6912)
                : m === "ffn_down" ? perModule(6912, 1536)
                : 0;
            bytes += v;
        }
        // 1 base + 2 Adam moments + 1 grad = 4x in-memory; on-disk is just 1x.
        return { mem: (bytes * 4 * n) / (1024 * 1024), disk: (bytes * n) / (1024 * 1024) };
    }, [lora]);

    // ─── Dataset handlers ──────────────────────────────────────────────

    const onFile = useCallback(async (f: File) => {
        const text = await f.text();
        const { examples: ex, errors } = parseJsonl(text);
        setDatasetName(f.name);
        setExamples(ex);
        setParseErrors(errors);
        setTokenLengths(null);
        if (ex.length > 0) {
            setPhase("ready");
            setAdapterName(f.name.replace(/\.[^.]+$/, "") + "-r" + lora.rank);
            toast.success(`Parsed ${ex.length} examples${errors.length ? `, ${errors.length} skipped` : ""}`);
        } else {
            setPhase("idle");
            toast.error("Couldn't parse any examples — check JSONL shape");
        }
    }, [toast, lora.rank]);

    const onValidate = useCallback(async () => {
        if (modelStatus !== "ready") {
            toast.error("Load a model in Chat first so I can tokenise");
            return;
        }
        try {
            const lens: number[] = [];
            for (const ex of examples) {
                const tokens = await client.encode(ex.prompt + ex.completion);
                lens.push(tokens.length);
            }
            setTokenLengths(lens);
            const over = lens.filter((n) => n > hp.max_seq_len).length;
            const msg = over > 0
                ? `${over} of ${lens.length} examples exceed seq_len cap of ${hp.max_seq_len} — they'll be truncated`
                : `All ${lens.length} examples fit under the seq_len cap`;
            toast.info(msg);
        } catch (e) {
            toast.error(`Validate failed: ${(e as Error).message}`);
        }
    }, [client, examples, hp.max_seq_len, modelStatus, toast]);

    // ─── Training drive ────────────────────────────────────────────────

    const runTraining = useCallback(async () => {
        if (modelStatus !== "ready" || examples.length === 0) return;
        cancelRef.current = false;
        setRecent([]);
        setErrorMsg(null);
        setPhase("training");
        // Cold-start affordance — show the `starting` progress phase
        // before the first beacon arrives. The 5 s watchdog flips on
        // a "compiling WGSL" hint, which is the actual reason the
        // first step is slow (pipeline cache cold on first dispatch).
        setProgress({ phase: "starting", current: 0, total: 0 });
        setColdHint(null);
        const coldTimer = window.setTimeout(() => {
            setColdHint("Compiling WGSL shaders — first step is slow on this device");
        }, 5000);

        let sessionStarted = false;
        let sessionAcquired = false;
        try {
            await client.acquireSession();
            sessionAcquired = true;
            // Clear any active adapter first so chat doesn't see stale weights
            // post-finish (training writes to a fresh LoraState).
            if (activeAdapter) {
                try { await client.trainingClearAdapter(); } catch { /* */ }
                onAdapterChanged?.(null);
            }
            // The worker probes the scratch+LoRA fit before consuming the
            // Model. If the device can't fit the requested config, this
            // throws with a "Training would need ~X MB…" message and the
            // Model stays alive in chat.
            await client.trainingStart({
                loraConfig: lora,
                hparams: hp,
                totalSteps: stepsBudget,
            });
            sessionStarted = true;

            for (let i = 0; i < stepsBudget; i++) {
                if (cancelRef.current) break;
                const ex = examples[i % examples.length];
                const text = ex.prompt + ex.completion;
                const allTokens = await client.encode(text);
                const promptTokens = await client.encode(ex.prompt);
                if (allTokens.length === 0 || promptTokens.length === 0) continue;
                const truncated = allTokens.slice(0, Math.max(2, hp.max_seq_len));
                // **A3 — UI safety net for NaN/Inf loss.** The worker
                // detects divergence first and throws; this catches
                // the case where the worker missed it (e.g., loss
                // came back as a poisoned f32 that JSON-serialized
                // to `null`). Throwing here funnels into the existing
                // catch which moves us to "error" phase cleanly.
                const checkLoss = (r: { loss: number; step: number }) => {
                    if (typeof r.loss !== "number" || !Number.isFinite(r.loss)) {
                        throw new Error(
                            `training diverged at step ${r.step} — loss is ${r.loss}. ` +
                            `Try a lower learning rate, smaller rank, or shorter seq_len.`,
                        );
                    }
                };
                if (hp.loss_mode === "per_position") {
                    const targets = new Uint32Array(truncated.length);
                    for (let p = 0; p < truncated.length; p++) {
                        const inPrompt = p < promptTokens.length - 1;
                        const hasNext = p + 1 < truncated.length;
                        targets[p] = (inPrompt || !hasNext) ? 0xFFFFFFFF : truncated[p + 1];
                    }
                    const t0 = performance.now();
                    const r = await client.trainingStep({
                        inputIds: truncated,
                        targets,
                        lossMode: "per_position",
                    });
                    checkLoss(r);
                    setRecent((prev) => [...prev.slice(-199), { ...r, ms: performance.now() - t0 }]);
                } else {
                    const promptOnly = truncated.slice(0, promptTokens.length);
                    const targetId = truncated[promptTokens.length] ?? truncated[truncated.length - 1];
                    const t0 = performance.now();
                    const r = await client.trainingStep({
                        inputIds: promptOnly.length > 0 ? promptOnly : truncated.slice(0, -1),
                        targetId,
                        lossMode: "next_token",
                    });
                    checkLoss(r);
                    setRecent((prev) => [...prev.slice(-199), { ...r, ms: performance.now() - t0 }]);
                }
            }

            // Loop exited cleanly. Either ran all `stepsBudget`
            // iterations, or `cancelRef.current` was set between
            // steps and the loop broke; in both cases the user has
            // a partial-or-complete adapter to save.
            setPhase(recent.length > 0 ? "done" : "ready");
        } catch (e) {
            const msg = (e as Error).message ?? String(e);
            // Per-layer cancellation surfaces as a thrown "cancelled
            // by caller" error from the wasm side. If the user
            // explicitly clicked Cancel (cancelRef is true), treat it
            // as a graceful stop, not an error — let them choose
            // Save / Apply / Discard against the partial adapter.
            const userCancelled = cancelRef.current
                && /cancelled/i.test(msg);
            if (userCancelled) {
                setPhase(recent.length > 0 ? "done" : "ready");
            } else if (!sessionStarted) {
                // Probe-failure recovery: trainingStart threw before
                // consuming the Model. Drop back to "ready" so the
                // form stays editable.
                setErrorMsg(msg);
                setPhase("ready");
                if (sessionAcquired) {
                    try { await client.releaseSession(); } catch { /* */ }
                }
                toast.error(msg);
            } else {
                // Mid-training failure — keep "error" phase so the
                // user has a clear stop point and the Reset button.
                setErrorMsg(msg);
                setPhase("error");
                toast.error(`Training stopped: ${msg}`);
            }
        } finally {
            window.clearTimeout(coldTimer);
            setColdHint(null);
            setProgress(null);
        }
    }, [activeAdapter, client, examples, hp, lora, modelStatus, onAdapterChanged, recent.length, stepsBudget, toast]);

    const onStop = useCallback(() => {
        // Two-stage cancel:
        // 1. Set the JS-side flag so the driver loop in `runTraining`
        //    exits *after* the in-flight step resolves.
        // 2. Flip the GPU-side flag so the in-flight forward/backward
        //    layer walk bails at the next encoder boundary — bounded
        //    latency ~one layer (300 ms - 1 s on browser) instead of
        //    one full step (10-30 s).
        cancelRef.current = true;
        setPhase("stopping");
        client.trainingCancel().catch(() => { /* no session — fine */ });
    }, [client]);

    const onSave = useCallback(async () => {
        const name = adapterName.trim();
        if (!name) {
            toast.error("Give the adapter a name first");
            return;
        }
        // **D2 — collision warning.** trainingSaveAdapter silently
        // overwrites because OPFS `getFileHandle(..., {create: true})`
        // is a no-prompt clobber. Surfaces here instead so the user
        // doesn't lose a prior adapter to a duplicate name.
        if (adapters.some((a) => a.name === name)) {
            const ok = window.confirm(`Overwrite existing adapter "${name}"? The previous version will be permanently lost.`);
            if (!ok) return;
        }
        try {
            const r = await client.trainingSaveAdapter(name);
            toast.success(`Saved ${r.name}.bin (${formatBytes(r.size)})`);
            await refreshAdapters();
        } catch (e) {
            toast.error(`Save failed: ${(e as Error).message}`);
        }
    }, [adapterName, adapters, client, refreshAdapters, toast]);

    const onFinishAndApply = useCallback(async () => {
        // **D4 — save-then-finish ordering.** Previously the order was
        // save → finish → apply. If save threw, finish() had already
        // run on the wrong side of the catch in a way that lost the
        // in-memory adapter forever. Now: save FIRST (only proceed if
        // it succeeds); finish + apply only after a successful save.
        // On save failure, leave the training session active so the
        // user can retry save with a different name / different
        // storage.
        const name = adapterName.trim();
        if (name && adapters.some((a) => a.name === name)) {
            const ok = window.confirm(`Overwrite existing adapter "${name}"? The previous version will be permanently lost.`);
            if (!ok) return;
        }
        try {
            if (name) {
                // Save first. If this throws, the training session is
                // still alive — user can rename and retry, or click
                // Discard to release the Model back to chat.
                await client.trainingSaveAdapter(name);
            }
            // Only finish + apply after save succeeded (or no name).
            await client.trainingFinish();
            if (name) {
                await client.trainingApplyAdapter(name);
                onAdapterChanged?.(name);
                toast.success(`Adapter "${name}" applied to chat`);
            } else {
                toast.info("Training finished — adapter not saved");
            }
            setPhase("idle");
            setRecent([]);
            setExamples([]);
            setDatasetName(null);
            await refreshAdapters();
            try { await client.releaseSession(); }
            catch (e) { console.warn("[fine-tune] releaseSession after finish failed:", e); }
        } catch (e) {
            toast.error(`Finish failed: ${(e as Error).message}`);
        }
    }, [adapterName, adapters, client, onAdapterChanged, refreshAdapters, toast]);

    const onDiscard = useCallback(async () => {
        // Best-effort: release any active training session + session
        // lock. Both can throw if there isn't one (e.g. probe-failure
        // path where Model was never consumed); swallow.
        try { await client.trainingFinish(); } catch { /* */ }
        try { await client.releaseSession(); } catch { /* */ }
        // Preserve the loaded dataset if there is one — user just hit
        // an error mid-training or wants to retry. Going all the way
        // back to "idle" forces them to re-pick the file, which is
        // exactly the kind of friction the W4 hardening is supposed
        // to avoid.
        setPhase(examples.length > 0 ? "ready" : "idle");
        setRecent([]);
        setErrorMsg(null);
    }, [client, examples.length]);

    // ─── Render ────────────────────────────────────────────────────────

    if (modelStatus !== "ready") {
        return (
            <div className="mx-auto mt-6 w-full max-w-md">
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-base">
                            <Sparkles className="size-4" /> Fine-tune
                        </CardTitle>
                        <CardDescription>
                            Load a model from the Chat tab first. Fine-tuning runs against
                            the same Model handle — no second load needed.
                        </CardDescription>
                    </CardHeader>
                </Card>
            </div>
        );
    }

    const isTraining = phase === "training" || phase === "stopping";
    const lastLoss = recent.length > 0 ? recent[recent.length - 1].loss : null;
    const firstLoss = recent.length > 0 ? recent[0].loss : null;
    const lossDirection = recent.length >= 5
        ? recent.slice(-5).reduce((a, b) => a + b.loss, 0) / 5
            - recent.slice(-10, -5).reduce((a, b) => a + b.loss, 0) / 5
        : 0;
    const progressPct = isTraining
        ? Math.min(100, (recent.length / Math.max(1, stepsBudget)) * 100)
        : 0;

    return (
        <div className="flex h-full min-h-0 flex-col overflow-hidden">
            {/* Status banner: which adapter is active in chat. */}
            {activeAdapter && (
                <div className="flex items-center gap-2 border-b border-border bg-card/30 px-4 py-2 text-xs text-muted-foreground">
                    <Plug className="size-3" />
                    Chat is using adapter
                    <Badge tone="muted" className="text-[10px]">{activeAdapter}</Badge>
                </div>
            )}

            <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto p-4 md:grid-cols-[1fr_320px]">
                {/* ─── Left: dataset workspace + live training panel ─── */}
                <div className="flex min-w-0 flex-col gap-4">
                    {!isTraining && phase !== "done" && (
                        <DatasetCard
                            datasetName={datasetName}
                            examples={examples}
                            parseErrors={parseErrors}
                            tokenLengths={tokenLengths}
                            seqCap={hp.max_seq_len}
                            onFile={onFile}
                            onValidate={onValidate}
                        />
                    )}

                    {(isTraining || phase === "done" || phase === "error" || recent.length > 0) && (
                        <LivePanel
                            phase={phase}
                            stepsBudget={stepsBudget}
                            recent={recent}
                            firstLoss={firstLoss}
                            lastLoss={lastLoss}
                            lossDirection={lossDirection}
                            progressPct={progressPct}
                            errorMsg={errorMsg}
                            progress={isTraining ? progress : null}
                            coldHint={isTraining ? coldHint : null}
                        />
                    )}
                </div>

                {/* ─── Right: hyperparams + actions ─── */}
                <div className="flex min-w-0 flex-col gap-4">
                    <MemoryTightToggle
                        on={memoryTight}
                        onChange={applyMemoryTight}
                        disabled={isTraining}
                    />
                    <ObjectiveCard
                        hp={hp} setHp={setHp}
                        stepsBudget={stepsBudget} setStepsBudget={setStepsBudget}
                        disabled={isTraining}
                        seqLenLocked={memoryTight}
                    />
                    <LoraShapeCard
                        lora={lora} setLora={setLora}
                        estimated={estimatedAdapterMB}
                        disabled={isTraining || memoryTight}
                    />
                    <AdvancedCard
                        hp={hp} setHp={setHp}
                        disabled={isTraining}
                    />
                </div>
            </div>

            {/* ─── Sticky action bar ─── */}
            <div className="flex shrink-0 flex-wrap items-center gap-2 border-t border-border bg-card/50 px-4 py-3">
                {phase === "ready" && (
                    <>
                        <Button onClick={runTraining} className="gap-1">
                            <Play className="size-4" /> Start training
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => { setExamples([]); setDatasetName(null); setPhase("idle"); setErrorMsg(null); }}>
                            Discard dataset
                        </Button>
                        {errorMsg && (
                            <span className="ml-2 flex-1 text-xs text-destructive">
                                {errorMsg}
                            </span>
                        )}
                    </>
                )}
                {phase === "idle" && examples.length === 0 && (
                    <span className="text-xs text-muted-foreground">
                        Drop a JSONL file above to begin.
                    </span>
                )}
                {(phase === "training" || phase === "stopping") && (
                    <>
                        <Button variant="destructive" size="sm" onClick={onStop} disabled={phase === "stopping"}>
                            <Square className="size-4" /> {phase === "stopping" ? "Stopping…" : "Cancel"}
                        </Button>
                        <div className="ml-auto text-xs text-muted-foreground">
                            step {recent.length} / {stepsBudget}
                        </div>
                    </>
                )}
                {phase === "done" && (
                    <>
                        <Input
                            value={adapterName}
                            onChange={(e) => setAdapterName(e.target.value)}
                            placeholder="adapter name"
                            className="max-w-xs"
                        />
                        <Button onClick={onSave} variant="secondary" className="gap-1">
                            <Save className="size-4" /> Save
                        </Button>
                        <Button onClick={onFinishAndApply} className="gap-1">
                            <CheckCircle2 className="size-4" /> Save + apply to chat
                        </Button>
                        <Button onClick={onDiscard} variant="ghost" size="sm">
                            Discard
                        </Button>
                    </>
                )}
                {phase === "error" && (
                    <>
                        <Button onClick={onDiscard} variant="secondary" size="sm" className="gap-1">
                            <RefreshCw className="size-3.5" /> Reset to retry
                        </Button>
                        <span className="ml-2 flex-1 text-xs text-destructive">{errorMsg}</span>
                    </>
                )}
            </div>

            {/* ─── Adapter library ─── */}
            {adapters.length > 0 && phase !== "training" && (
                <AdapterLibrary
                    entries={adapters}
                    active={activeAdapter}
                    onApply={async (n) => {
                        try {
                            await client.trainingApplyAdapter(n);
                            onAdapterChanged?.(n);
                            toast.success(`Applied "${n}"`);
                        } catch (e) { toast.error(`${(e as Error).message}`); }
                    }}
                    onClear={async () => {
                        try {
                            await client.trainingClearAdapter();
                            onAdapterChanged?.(null);
                            toast.success("Cleared adapter");
                        } catch (e) { toast.error(`${(e as Error).message}`); }
                    }}
                    onDelete={async (n) => {
                        if (!window.confirm(`Delete adapter "${n}"?`)) return;
                        try {
                            await client.trainingDeleteAdapter(n);
                            await refreshAdapters();
                            toast.info(`Deleted "${n}"`);
                        } catch (e) { toast.error(`${(e as Error).message}`); }
                    }}
                />
            )}
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────
// Subcomponents
// ────────────────────────────────────────────────────────────────────────

function DatasetCard(props: {
    datasetName: string | null;
    examples: ParsedExample[];
    parseErrors: string[];
    tokenLengths: number[] | null;
    seqCap: number;
    onFile: (f: File) => void;
    onValidate: () => void;
}) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);
    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f) props.onFile(f);
    };
    const histogram = useMemo(() => {
        if (!props.tokenLengths || props.tokenLengths.length === 0) return null;
        const max = Math.max(...props.tokenLengths);
        const bins = 12;
        const buckets = new Array(bins).fill(0);
        for (const n of props.tokenLengths) {
            const idx = Math.min(bins - 1, Math.floor((n / max) * bins));
            buckets[idx]++;
        }
        const peak = Math.max(...buckets);
        return buckets.map((c, i) => {
            const binMax = ((i + 1) * max) / bins;
            return { count: c, height: peak > 0 ? c / peak : 0, over: binMax > props.seqCap };
        });
    }, [props.tokenLengths, props.seqCap]);

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2"><FileText className="size-4" /> Dataset</CardTitle>
                <CardDescription>
                    JSONL with <code className="text-foreground">{"{prompt, completion}"}</code>{" "}
                    or Alpaca <code className="text-foreground">{"{instruction, output}"}</code> shape.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
                <div
                    role="button"
                    tabIndex={0}
                    onClick={() => inputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={onDrop}
                    className={cn(
                        "flex h-24 cursor-pointer items-center justify-center rounded-md border-2 border-dashed text-xs transition-colors",
                        dragOver ? "border-primary bg-primary/5" : "border-border text-muted-foreground hover:bg-muted/30",
                    )}
                >
                    {props.datasetName
                        ? <span className="text-foreground">{props.datasetName} — {props.examples.length} examples</span>
                        : <span>Drop a .jsonl file, or click to pick</span>}
                </div>
                <input
                    ref={inputRef}
                    type="file"
                    accept=".jsonl,.json,.txt"
                    className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) props.onFile(f); }}
                />
                {props.parseErrors.length > 0 && (
                    <div className="rounded border border-amber-500/30 bg-amber-500/5 p-2 text-xs text-amber-700 dark:text-amber-300">
                        <div className="font-medium">Skipped {props.parseErrors.length} malformed lines:</div>
                        {props.parseErrors.slice(0, 3).map((e, i) => (
                            <div key={i} className="truncate">• {e}</div>
                        ))}
                        {props.parseErrors.length > 3 && <div>…and {props.parseErrors.length - 3} more</div>}
                    </div>
                )}
                {props.examples.length > 0 && (
                    <>
                        <Button size="sm" variant="secondary" onClick={props.onValidate} className="gap-1">
                            <Activity className="size-3" /> Tokenise + validate
                        </Button>
                        {histogram && (
                            <div className="space-y-1">
                                <div className="text-xs text-muted-foreground">
                                    Token-length distribution (red = exceeds seq_len cap of {props.seqCap})
                                </div>
                                <div className="flex h-12 items-end gap-px rounded bg-muted/30 p-1">
                                    {histogram.map((b, i) => (
                                        <div
                                            key={i}
                                            className={cn(
                                                "flex-1 rounded-sm",
                                                b.over ? "bg-destructive/70" : "bg-primary/60",
                                            )}
                                            style={{ height: `${Math.max(4, b.height * 100)}%` }}
                                            title={`${b.count} examples`}
                                        />
                                    ))}
                                </div>
                            </div>
                        )}
                        <div className="max-h-32 overflow-y-auto rounded border border-border bg-muted/20 p-2 font-mono text-[11px] leading-snug">
                            {props.examples.slice(0, 5).map((ex, i) => (
                                <div key={i} className="mb-2 last:mb-0">
                                    <span className="text-muted-foreground">{ex.prompt}</span>
                                    <span className="text-foreground">{ex.completion}</span>
                                </div>
                            ))}
                            {props.examples.length > 5 && (
                                <div className="text-muted-foreground">…and {props.examples.length - 5} more</div>
                            )}
                        </div>
                    </>
                )}
            </CardContent>
        </Card>
    );
}

/** Memory-tight preset toggle. Renders as a compact card-shaped switch
 *  above ObjectiveCard / LoraShapeCard. When on:
 *   - LoraShapeCard sliders + target-module toggles are disabled.
 *   - ObjectiveCard's loss-mode picker + max_seq_len input are disabled.
 *   - The values are force-applied via `applyMemoryTight` in the parent.
 *
 *  Style mirrors the Card pattern used by ObjectiveCard / LoraShapeCard,
 *  with a clear "preset" affordance (the small description explains what
 *  it actually does so the user isn't guessing).
 */
function MemoryTightToggle(props: {
    on: boolean;
    onChange: (next: boolean) => void;
    disabled: boolean;
}) {
    return (
        <Card>
            {/* `py-5` (20 px symmetric) gives the checkbox the same
                vertical breathing room a `<CardHeader>` would, without
                imposing a full header element. CardContent's default
                p-6 assumes there's a CardHeader above — without one,
                the content lands flush against the card border. */}
            <CardContent className="py-5">
                <label className={cn(
                    "flex cursor-pointer items-start gap-3",
                    props.disabled && "cursor-not-allowed opacity-60",
                )}>
                    <input
                        type="checkbox"
                        checked={props.on}
                        onChange={(e) => props.onChange(e.target.checked)}
                        disabled={props.disabled}
                        className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-primary"
                    />
                    <div className="min-w-0 flex-1">
                        <div className="text-sm font-medium">Memory-tight (iPhone-safe) preset</div>
                        <div className="mt-0.5 text-xs text-muted-foreground">
                            Smallest config that's expected to fit on iPhone 16e:
                            rank&nbsp;1, alpha&nbsp;2, attn_q + attn_v only, seq_len&nbsp;16,
                            next-token loss, gradient checkpointing. Locks the LoRA shape
                            and max&nbsp;seq_len so the preset can't drift.
                        </div>
                    </div>
                </label>
            </CardContent>
        </Card>
    );
}

function ObjectiveCard(props: {
    hp: TrainingHyperparams; setHp: (h: TrainingHyperparams) => void;
    stepsBudget: number; setStepsBudget: (n: number) => void;
    disabled: boolean;
    /** When true (memory-tight preset is on), the max_seq_len + loss_mode
     *  controls are forced by the preset and shouldn't be editable. The
     *  rest of the card (steps, learning rate) stays interactive. */
    seqLenLocked?: boolean;
}) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2"><Settings2 className="size-4" /> Objective</CardTitle>
                <CardDescription>How the loss is computed.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
                <div className="flex gap-1 rounded-md border border-border bg-muted/30 p-0.5">
                    {(["next_token", "per_position"] as const).map((m) => (
                        <button
                            key={m}
                            disabled={props.disabled || !!props.seqLenLocked}
                            onClick={() => props.setHp({ ...props.hp, loss_mode: m })}
                            className={cn(
                                "flex-1 rounded px-2 py-1 text-xs transition-colors",
                                props.hp.loss_mode === m
                                    ? "bg-background text-foreground shadow-sm"
                                    : "text-muted-foreground hover:bg-background/50",
                                props.disabled && "cursor-not-allowed opacity-50",
                            )}
                        >
                            {m === "next_token" ? "Next-token" : "Per-position"}
                        </button>
                    ))}
                </div>
                <LabeledInput
                    label="Steps"
                    description="optimizer steps total"
                    value={props.stepsBudget}
                    min={1} max={10_000} step={10}
                    onChange={(n) => props.setStepsBudget(clampInt(n, 1, 10_000, 100))}
                    disabled={props.disabled}
                />
                <LabeledSlider
                    label="Learning rate"
                    valueLabel={`${props.hp.learning_rate.toExponential(2)}`}
                    value={Math.log10(props.hp.learning_rate)}
                    min={-5} max={-2} step={0.1}
                    onChange={(v) => props.setHp({ ...props.hp, learning_rate: Math.pow(10, v) })}
                    disabled={props.disabled}
                />
                <LabeledInput
                    label="Max seq_len"
                    description={`~${Math.round(props.hp.max_seq_len * 1.4)} MB GPU per layer × 35 layers`}
                    value={props.hp.max_seq_len}
                    min={16} max={2048} step={16}
                    onChange={(n) => props.setHp({ ...props.hp, max_seq_len: clampInt(n, 16, 2048, 128) })}
                    disabled={props.disabled || !!props.seqLenLocked}
                />
            </CardContent>
        </Card>
    );
}

function LoraShapeCard(props: {
    lora: TrainingLoraConfig; setLora: (l: TrainingLoraConfig) => void;
    estimated: { mem: number; disk: number };
    disabled: boolean;
}) {
    const toggleTarget = (m: string) => {
        const has = props.lora.target_modules.includes(m);
        const next = has
            ? props.lora.target_modules.filter((x) => x !== m)
            : [...props.lora.target_modules, m];
        props.setLora({ ...props.lora, target_modules: next });
    };
    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2"><Sparkles className="size-4" /> LoRA shape</CardTitle>
                <CardDescription>
                    Rank + which projections to wrap. Estimated{" "}
                    <span className="text-foreground">{props.estimated.mem.toFixed(1)} MB</span>{" "}
                    in GPU memory,{" "}
                    <span className="text-foreground">{props.estimated.disk.toFixed(1)} MB</span>{" "}
                    on disk.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
                <LabeledSlider
                    label="Rank"
                    valueLabel={`r=${props.lora.rank}`}
                    value={props.lora.rank}
                    min={1} max={32} step={1}
                    onChange={(v) => props.setLora({ ...props.lora, rank: clampInt(v, 1, 32, 4) })}
                    disabled={props.disabled}
                />
                <LabeledSlider
                    label="Alpha"
                    valueLabel={`α=${props.lora.alpha}`}
                    value={props.lora.alpha}
                    min={1} max={64} step={1}
                    onChange={(v) => props.setLora({ ...props.lora, alpha: clampNum(v, 1, 64, 8) })}
                    disabled={props.disabled}
                />
                <div>
                    <div className="mb-1 text-xs text-muted-foreground">Target modules</div>
                    <div className="flex flex-wrap gap-1">
                        {ALL_TARGETS.map((m) => {
                            const on = props.lora.target_modules.includes(m);
                            return (
                                <button
                                    key={m}
                                    type="button"
                                    onClick={() => !props.disabled && toggleTarget(m)}
                                    disabled={props.disabled}
                                    className={cn(
                                        "rounded border px-2 py-0.5 text-[11px] transition-colors",
                                        on
                                            ? "border-primary/50 bg-primary/10 text-primary"
                                            : "border-border text-muted-foreground hover:bg-muted/30",
                                        props.disabled && "cursor-not-allowed opacity-50",
                                    )}
                                >
                                    {m}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

function AdvancedCard(props: {
    hp: TrainingHyperparams; setHp: (h: TrainingHyperparams) => void;
    disabled: boolean;
}) {
    const [open, setOpen] = useState(false);
    return (
        <Card>
            <CardHeader>
                <button
                    type="button"
                    className="flex w-full items-center justify-between"
                    onClick={() => setOpen(!open)}
                >
                    <div className="text-left">
                        <CardTitle>Advanced</CardTitle>
                        <CardDescription>Schedule, clipping, checkpointing, seed.</CardDescription>
                    </div>
                    <RefreshCw className={cn("size-4 transition-transform", open && "rotate-180")} />
                </button>
            </CardHeader>
            {open && (
                <CardContent className="space-y-3">
                    <LabeledInput
                        label="Warmup steps" description=""
                        value={props.hp.warmup_steps}
                        min={0} max={1000} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, warmup_steps: clampInt(n, 0, 1000, 0) })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Grad accum" description="micro-batches per optimizer step"
                        value={props.hp.gradient_accumulation_steps}
                        min={1} max={32} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, gradient_accumulation_steps: clampInt(n, 1, 32, 1) })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Grad clip" description="L2 norm, 0 = off"
                        value={props.hp.max_grad_norm}
                        min={0} max={10} step={0.1}
                        onChange={(n) => props.setHp({ ...props.hp, max_grad_norm: clampNum(n, 0, 10, 0) })}
                        disabled={props.disabled}
                    />
                    <LabeledToggle
                        label="Gradient checkpointing"
                        description="trade compute for memory"
                        value={props.hp.gradient_checkpointing}
                        onChange={(v) => props.setHp({ ...props.hp, gradient_checkpointing: v })}
                        disabled={props.disabled}
                    />
                    <LabeledToggle
                        label="Mixed precision"
                        description="f16 adapter on disk"
                        value={props.hp.mixed_precision}
                        onChange={(v) => props.setHp({ ...props.hp, mixed_precision: v })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Backward layer floor"
                        description="0 = train every layer; N = only train layers ≥ N (smaller adapter, less GPU memory)"
                        value={props.hp.backward_layer_floor ?? 0}
                        min={0} max={34} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, backward_layer_floor: clampInt(n, 0, 34, 0) })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Seed" description="LoRA A init reproducibility"
                        value={props.hp.seed}
                        min={0} max={2 ** 31 - 1} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, seed: clampInt(n, 0, 2 ** 31 - 1, 42) })}
                        disabled={props.disabled}
                    />
                </CardContent>
            )}
        </Card>
    );
}

function LivePanel(props: {
    phase: Phase;
    stepsBudget: number;
    recent: RecentStep[];
    firstLoss: number | null;
    lastLoss: number | null;
    lossDirection: number;
    progressPct: number;
    errorMsg: string | null;
    progress: TrainingProgressState | null;
    coldHint: string | null;
}) {
    const dropPct = (props.firstLoss && props.lastLoss != null)
        ? ((props.firstLoss - props.lastLoss) / Math.max(props.firstLoss, 1e-6)) * 100
        : 0;
    const meanMs = props.recent.length > 0
        ? props.recent.reduce((a, b) => a + b.ms, 0) / props.recent.length
        : 0;
    const eta = props.phase === "training" && meanMs > 0
        ? Math.max(0, (props.stepsBudget - props.recent.length) * meanMs) / 1000
        : 0;
    return (
        <Card>
            <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                    <Activity className="size-4" />
                    {props.phase === "done" ? "Training complete" :
                     props.phase === "error" ? "Training stopped" :
                     props.phase === "stopping" ? "Cancelling…" :
                     "Live training"}
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
                {props.errorMsg && (
                    <div className="flex items-start gap-2 rounded border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
                        <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                        <span className="break-words">{props.errorMsg}</span>
                    </div>
                )}
                {/* Mirrors PipelineProgress for chat: real-time beacon
                    from inside the wasm trainer — phase + per-layer or
                    per-token tick — so the user can see exactly what
                    the GPU is doing during the otherwise-silent
                    multi-second step. */}
                {props.progress && (
                    <TrainingProgress
                        state={props.progress}
                        stepBudget={props.stepsBudget}
                        coldHint={props.coldHint ?? undefined}
                    />
                )}
                <div className="flex items-baseline gap-3">
                    <span
                        className="text-4xl font-semibold tabular-nums tracking-tight"
                        aria-live="polite"
                    >
                        {props.lastLoss != null ? props.lastLoss.toFixed(4) : "—"}
                    </span>
                    {props.lastLoss != null && (
                        <Badge tone="muted" className={cn(
                            "text-[10px]",
                            props.lossDirection < 0 ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400"
                            : props.lossDirection > 0 ? "border-amber-500/40 text-amber-600 dark:text-amber-400"
                            : ""
                        )}>
                            {props.lossDirection < 0 ? "▼ falling" : props.lossDirection > 0 ? "▲ spiking" : "flat"}
                        </Badge>
                    )}
                    <span className="ml-auto text-xs text-muted-foreground">
                        {props.firstLoss != null && `from ${props.firstLoss.toFixed(2)} (${dropPct.toFixed(0)}% drop)`}
                    </span>
                </div>
                <Progress value={props.progressPct} className="h-1" />
                <div className="flex flex-wrap gap-x-4 text-xs text-muted-foreground">
                    <span>step <span className="text-foreground">{props.recent.length}</span> / {props.stepsBudget}</span>
                    {meanMs > 0 && <span>~<span className="text-foreground">{meanMs.toFixed(0)} ms</span>/step</span>}
                    {eta > 0 && <span>ETA <span className="text-foreground">{formatDuration(eta)}</span></span>}
                </div>
                {props.recent.length >= 2 && <Sparkline data={props.recent.map((s) => s.loss)} />}
                {props.recent.length >= 2 && (
                    <div className="text-xs text-muted-foreground">Last 20 step latencies:</div>
                )}
                {props.recent.length >= 2 && (
                    <LatencyStrip data={props.recent.slice(-20).map((s) => s.ms)} />
                )}
            </CardContent>
        </Card>
    );
}

function Sparkline({ data }: { data: number[] }) {
    const w = 320, h = 60;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const points = data.map((v, i) => {
        const x = (i / Math.max(1, data.length - 1)) * w;
        const y = h - ((v - min) / range) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return (
        <svg viewBox={`0 0 ${w} ${h}`} className="h-16 w-full rounded bg-muted/20" preserveAspectRatio="none">
            <polyline
                fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary"
                points={points.join(" ")}
            />
        </svg>
    );
}

function LatencyStrip({ data }: { data: number[] }) {
    const max = Math.max(...data, 1);
    return (
        <div className="flex h-6 items-end gap-px rounded bg-muted/20 p-0.5">
            {data.map((ms, i) => (
                <div
                    key={i}
                    className="flex-1 rounded-sm bg-muted-foreground/40"
                    style={{ height: `${(ms / max) * 100}%` }}
                    title={`${ms.toFixed(0)} ms`}
                />
            ))}
        </div>
    );
}

function AdapterLibrary(props: {
    entries: AdapterListEntry[];
    active: string | null;
    onApply: (n: string) => void;
    onClear: () => void;
    onDelete: (n: string) => void;
}) {
    return (
        <div className="shrink-0 border-t border-border bg-card/30 p-4">
            <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Saved adapters</span>
                {props.active && (
                    <Button size="sm" variant="ghost" onClick={props.onClear} className="h-6 gap-1 px-2 text-xs">
                        <Plug className="size-3" /> Clear active
                    </Button>
                )}
            </div>
            <div className="flex flex-wrap gap-2">
                {props.entries.map((e) => {
                    const isActive = props.active === e.name;
                    return (
                        <Card key={e.name} className={cn("p-2", isActive && "border-primary/50 bg-primary/5")}>
                            <div className="flex items-center gap-2">
                                <div className="min-w-0">
                                    <div className="truncate text-xs font-medium">
                                        {e.name}
                                        {isActive && <Badge tone="muted" className="ml-1 text-[10px]">active</Badge>}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground">
                                        {formatBytes(e.size)} · {new Date(e.lastModified).toLocaleDateString()}
                                    </div>
                                </div>
                                <div className="ml-auto flex gap-0.5">
                                    {!isActive && (
                                        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => props.onApply(e.name)}>
                                            Apply
                                        </Button>
                                    )}
                                    <Button size="sm" variant="ghost" className="h-6 px-1 text-destructive" onClick={() => props.onDelete(e.name)}>
                                        <Trash2 className="size-3" />
                                    </Button>
                                </div>
                            </div>
                        </Card>
                    );
                })}
            </div>
        </div>
    );
}

// ─── Form primitives ─────────────────────────────────────────────────────

function LabeledInput(props: {
    label: string; description: string;
    value: number; min: number; max: number; step: number;
    onChange: (n: number) => void;
    disabled: boolean;
}) {
    return (
        <div className="space-y-1">
            <label className="flex items-baseline justify-between text-xs">
                <span>{props.label}</span>
                {props.description && <span className="text-muted-foreground">{props.description}</span>}
            </label>
            <Input
                type="number"
                min={props.min} max={props.max} step={props.step}
                value={props.value}
                onChange={(e) => props.onChange(Number(e.target.value))}
                disabled={props.disabled}
                className="h-8"
            />
        </div>
    );
}

function LabeledSlider(props: {
    label: string; valueLabel: string;
    value: number; min: number; max: number; step: number;
    onChange: (n: number) => void;
    disabled: boolean;
}) {
    return (
        <div className="space-y-1">
            <label className="flex items-baseline justify-between text-xs">
                <span>{props.label}</span>
                <span className="text-foreground tabular-nums">{props.valueLabel}</span>
            </label>
            <input
                type="range"
                value={props.value}
                min={props.min} max={props.max} step={props.step}
                onChange={(e) => props.onChange(Number(e.target.value))}
                disabled={props.disabled}
                className={cn(
                    "h-5 w-full cursor-pointer appearance-none rounded-md bg-muted accent-primary",
                    props.disabled && "cursor-not-allowed opacity-50",
                )}
            />
        </div>
    );
}

function LabeledToggle(props: {
    label: string; description: string;
    value: boolean;
    onChange: (v: boolean) => void;
    disabled: boolean;
}) {
    return (
        <label className={cn(
            "flex cursor-pointer items-center justify-between rounded border border-border px-2 py-1 text-xs",
            props.disabled && "cursor-not-allowed opacity-50",
        )}>
            <div>
                <div>{props.label}</div>
                <div className="text-muted-foreground">{props.description}</div>
            </div>
            <input
                type="checkbox"
                checked={props.value}
                onChange={(e) => props.onChange(e.target.checked)}
                disabled={props.disabled}
                className="size-4"
            />
        </label>
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}
