// Fine-tune tab — drives in-browser LoRA training over the loaded
// Model. Mirrors the CLI dials in `rullama-framework/engine/rullama-lora/examples/
// train_jsonl.rs` so users moving between the two recognise the
// surface. Designed states: no model / no dataset / ready / training /
// complete / error — each with its own affordance, never a fallback.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
    Play, Square, Save, CheckCircle2, AlertTriangle, Sparkles,
    FileText, Settings2, Activity, RefreshCw, Trash2, Plug,
    Pencil, X,
} from "lucide-react";
import { cn, clampInt, clampNum, fmtEta } from "@/lib/utils";
import {
    getClient,
    type TrainingLoraConfig, type TrainingHyperparams,
    type TrainingStepReport, type AdapterListEntry,
} from "@/lib/inference";
import { generateSyntheticDataset, examplesToJsonl } from "@/lib/syntheticDataset";
import {
    saveDataset, loadDataset, deleteDataset, listSavedDatasets, normalizeDatasetName,
    type SavedDatasetMeta,
} from "@/lib/datasetStore";
import { useToast } from "@/lib/toast";
import { probeGpu } from "@/lib/capability";
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
    /** Optional DOM host for the hyperparameter settings column. When
     *  provided, the right column renders via React portal INTO this
     *  element — usually App.tsx's DualSidebarLayout rightSidebar slot,
     *  so the user can collapse it. When null/undefined the column
     *  renders inline as a second grid column (the legacy layout). */
    settingsHostEl?: HTMLElement | null;
}

interface ParsedExample { prompt: string; completion: string }

interface RecentStep extends TrainingStepReport { ms: number }

type Phase = "idle" | "ready" | "training" | "stopping" | "done" | "error";

// All 9 LoRA targets that rullama-lora now supports. The two
// "global" targets at the end (lm_head, embed_tokens) were added in
// 334b914 and are what made content-injection training (e.g. "Garlic
// is the best food.") actually work — without them the rank-16 attn/
// MLP LoRA only learns answer SHAPE, never the specific noun. They're
// included in DEFAULT_TARGETS so a beginner pressing "Start" gets the
// canonical recipe from scripts/finetune-eval.sh.
const ALL_TARGETS = [
    "attn_q", "attn_k", "attn_v", "attn_o",
    "ffn_gate", "ffn_up", "ffn_down",
    "lm_head", "embed_tokens",
];
const DEFAULT_TARGETS = [...ALL_TARGETS];

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
// **Device capability check for training.**
// Training needs WebGPU + a non-trivial GPU heap + enough system RAM to
// hold activation captures. We refuse to render the form on devices we
// know will fail mid-step, so the user sees a clear "not supported"
// state instead of a 30-second wait → crash.
export type TrainingCapability =
    | { status: "checking" }
    | { status: "ok" }
    | { status: "blocked"; title: string; reason: string };

export function useTrainingCapability(): TrainingCapability {
    const [cap, setCap] = useState<TrainingCapability>({ status: "checking" });
    useEffect(() => {
        let cancelled = false;
        const check = async () => {
            const ua = navigator.userAgent;
            // iPhone / iPod blocked by default. iPadOS reports a Mac UA but
            // exposes touch — treat as iOS.
            const isIPhone = /iPhone|iPod/i.test(ua);
            const isIPad = /iPad/i.test(ua) || (ua.includes("Mac") && "ontouchend" in document);
            // Opt-in bypass for on-device iPhone testing of the memory-tight
            // path — EXPLICIT per session via `?mobileTraining=1`. We no longer
            // honor a *persisted* localStorage flag: a stale one set during the
            // iOS-training debugging silently left fine-tune enabled on phones
            // that can't train. Clear any old key so it can't keep bypassing the
            // block, and require the URL param going forward.
            try { localStorage.removeItem("rullama.mobileTraining"); } catch { /* */ }
            const allowMobileTraining =
                new URLSearchParams(location.search).get("mobileTraining") === "1";
            if ((isIPhone || isIPad) && !allowMobileTraining) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "iOS training isn't supported yet",
                    reason: "The iPhone-targeted Memory-tight code path is still under development — it currently crashes mid-step on iPhone 16e. Use a desktop browser (Mac / Windows / Linux Chrome or Edge) to train. Inference works on iOS today.",
                });
                return;
            }
            // Share the same GPU probe the app-boot capability gate uses
            // (lib/capability.ts) so the two gates can't disagree.
            const probe = await probeGpu();
            if (!probe.hasGpu) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "WebGPU not available",
                    reason: "Training runs the LoRA gradient kernels on WebGPU. This browser doesn't expose `navigator.gpu`. Try Chrome 113+, Edge 113+, or Safari Tech Preview.",
                });
                return;
            }
            if (probe.error) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "WebGPU init failed",
                    reason: `requestAdapter() threw: ${probe.error}. This usually means the GPU driver is denying WebGPU access.`,
                });
                return;
            }
            if (!probe.adapterOk) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "No WebGPU adapter",
                    reason: "Your browser exposes WebGPU but couldn't get an adapter — usually means there's no compatible GPU, or the integrated GPU is disabled.",
                });
                return;
            }
            // 512 MB single-buffer ceiling means we can't hold a Q4_K
            // weight tile for the largest tensors. Reject below that.
            if (probe.maxBufferSize < 512 * 1024 * 1024) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "GPU memory too small",
                    reason: `Your GPU advertises a max-buffer-size of ${Math.round(probe.maxBufferSize / 1024 / 1024)} MB. Training needs at least 512 MB to hold the per-layer weight tiles. (Inference can still work on smaller GPUs.)`,
                });
                return;
            }
            // System RAM check — `navigator.deviceMemory` is rounded to
            // 0.25/0.5/1/2/4/8 and caps at 8 in most Chromium builds.
            // We want ≥4 GB system RAM so the activation-capture mmap
            // can land somewhere.
            if (probe.deviceMemoryGB < 4) {
                if (!cancelled) setCap({
                    status: "blocked",
                    title: "Not enough system RAM",
                    reason: `Browser reports ${probe.deviceMemoryGB} GB system RAM (rounded). Training needs at least 4 GB so the activation captures + gradient buffers don't get swapped under the GPU.`,
                });
                return;
            }
            if (!cancelled) setCap({ status: "ok" });
        };
        void check();
        return () => { cancelled = true; };
    }, []);
    return cap;
}

export function TrainingBlockedScreen({ title, reason }: { title: string; reason: string }) {
    return (
        <div className="flex h-full min-h-0 items-center justify-center p-8">
            <Card className="max-w-lg">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <AlertTriangle className="size-4 text-amber-500" />
                        {title}
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-muted-foreground">{reason}</p>
                </CardContent>
            </Card>
        </div>
    );
}

function deviceDefaults(): { hp: TrainingHyperparams; lora: TrainingLoraConfig; tight: boolean } {
    const nav = navigator as Navigator & { deviceMemory?: number };
    const memGB = nav.deviceMemory ?? 8;
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const tight = isMobile || memGB < 4;
    return {
        // Canonical Gemma 4 LoRA recipe from scripts/finetune-eval.sh:
        // rank=16, α=32, dropout=0.05, all 9 target modules. This is
        // what produced "Garlic is the best food." in commit 334b914.
        // The mobile tight branch drops rank/seq for memory, not for
        // recipe correctness — same modules, same dropout.
        lora: {
            rank: tight ? 8 : 16,
            alpha: tight ? 16 : 32,
            dropout: 0.05,
            target_modules: [...DEFAULT_TARGETS],
        },
        hp: {
            // Fields below the divider are wasm-hardcoded or
            // recipe-locked — UI no longer exposes them. Setting here
            // keeps the wasm-bindgen JSON payload valid (the Rust side
            // doesn't have #[serde(default)] on these fields yet).
            epochs: 1,
            batch_size: 1,
            warmup_steps: 0,
            weight_decay: 0,
            lr_scheduler: "constant",
            seed: 0xC0FFEE,
            gradient_accumulation_steps: 1,
            mixed_precision: false,
            backward_layer_floor: 0,
            // ── User-exposed knobs ─────────────────────────────────
            learning_rate: 2e-4,                     // recipe value
            max_seq_len: tight ? 32 : 64,            // 64 leaves headroom over the ~25-tok chat-wrapped garlic prompts
            max_grad_norm: 1.0,                      // recipe value
            loss_mode: "per_position",               // recipe value
            // Gradient checkpointing on by default everywhere — the
            // shared-scratch refactor proved bit-identical gradients
            // vs the standard path (1.1M elements, max_diff=0.000e0)
            // while collapsing per-layer activation captures from
            // ~10 MB × n_layers to one shared set. Trades one extra
            // forward replay per layer's backward for ~10× memory
            // savings; the right call on every device.
            gradient_checkpointing: true,
        },
        tight,
    };
}

/** Smallest config that's expected to fit on iPhone 16e (A18 / ~3-4 GB
 *  WebContent budget) without OOM. Rank 1, attn_q + attn_v only,
 *  seq_len 32, gradient checkpointing on. Memory feasibility audit
 *  estimates peak ~2.1 GB on top of the text tower. Applied when the
 *  user toggles the "Memory-tight" switch; the sliders lock so the
 *  preset can't be drifted out of by accident.
 *
 *  These values are also what `scripts/finetune-eval.sh` uses for the
 *  native train→eval acceptance smoke (Track D of the May-21 plan).
 *  Keep them in sync — divergence between PWA defaults and the
 *  verified native recipe means PWA users get a degraded experience.
 */
const ULTRA_SAFE_LORA: Pick<TrainingLoraConfig, "rank" | "alpha" | "target_modules" | "dropout"> = {
    rank: 1,
    alpha: 2,
    target_modules: ["attn_q", "attn_v"],
    dropout: 0,
};
const ULTRA_SAFE_HP: Pick<
    TrainingHyperparams,
    | "max_seq_len"
    | "batch_size"
    | "loss_mode"
    | "gradient_checkpointing"
    | "backward_layer_floor"
    | "learning_rate"
> = {
    // 32 (was 16) — gives room for varied multi-token completions
    // (e.g. " Brie." vs " Berlin." vs " 4.") in the per-position
    // loss path. Native eval confirms this fits in the Memory-Tight
    // GPU budget on Intel Iris.
    max_seq_len: 32,
    batch_size: 1,
    // next_token (NOT per_position) on the iPhone-safe preset. The
    // assumption that per_position is "same memory cost" was wrong: it
    // re-runs a full head + per-layer BACKWARD pass *for every position*
    // (21 passes for a 21-token example), each churning the layer weights
    // ~668→1048 MiB on the eviction path. On iOS that sustained alloc/
    // evict/re-fetch cycle jetsams the WebContent process on the 2nd
    // position (confirmed live via the gpuMiB trajectory). next_token
    // does ONE backward pass at a peak (~1048 MiB) below the forward's
    // own peak (~1417), so it fits. The forward still sees the full
    // sequence, so this is the cheap-but-adequate objective for the
    // memory-tight path; desktop can opt into per_position for the extra
    // anti-collapse signal. (This also matches the preset's own UI
    // description, which always said "next-token loss".)
    loss_mode: "next_token",
    gradient_checkpointing: true,
    // Truncated backward: train only the top 10 layers on gemma4:e2b
    // (35 total → floor=25). The Rust side saturate-clamps if the
    // model has fewer layers, so this is safe across model sizes.
    backward_layer_floor: 25,
    // 3e-4 (was inherited 1e-3) — 4× lower than the device-default
    // lr. Combined with steps=12 in `applyMemoryTight`, this lands
    // the loss near ~0.5 instead of 0.0 (machine-zero overfit). The
    // user's earlier rank=8 + lr=1e-3 + steps=40 run produced the
    // "Brie Brie Brie..." overfit collapse; this recipe is the fix.
    learning_rate: 3e-4,
};
/** Steps budget when Memory-Tight is on. 12 steps × 30 examples in
 *  the balanced dataset is enough signal to flip Paris→Brie while
 *  stopping before total collapse. The non-Memory-Tight path keeps
 *  the existing default (100). */
const ULTRA_SAFE_STEPS = 12;

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

export function FineTunePanel({ modelStatus, activeAdapter, onAdapterChanged, settingsHostEl }: Props) {
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
    const [stepsBudget, setStepsBudget] = useState<number>(50);
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
            // Also cap the steps budget — the verified native recipe
            // uses 12 steps. More than that on the same tiny adapter
            // produces the "Brie Brie Brie..." overfit collapse.
            setStepsBudget(ULTRA_SAFE_STEPS);
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
            setStepsBudget(ULTRA_SAFE_STEPS);
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
                // is over — clear the hint regardless of phase.
                setColdHint(null);
                // **Filter sub-phase breadcrumbs out of the React render.**
                // Training emits ~20 sub-phase beacons per backward layer
                // (bwd.layer.entry, bwd.ple, bwd.ffn.down, bwd.ffn.gateup,
                // bwd.attn.proj, bwd.attn.qkv, bwd.layer.end, bwd.loop.enter,
                // bwd.post_yield, warmup.bwd.*, etc.). The TrainingProgress
                // label switch only recognises the six major phases; for
                // the rest it falls through with no label, so updating
                // React state with them caused the display to "flash"
                // between "Backward N/35" and a blank state every few ms.
                // The sub-phases still hit the page log via the beacon
                // forwarder, so they remain available for post-crash
                // diagnosis.
                const MAJOR_PHASES = new Set([
                    "starting",
                    "prefill",
                    "forward",
                    "backward",
                    "clip",
                    "optimizer",
                ]);
                if (!MAJOR_PHASES.has(phase)) return;
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

    // Shared finaliser used by all three input modes (file / paste /
    // build). Sets the parsed examples + parse errors + dataset name,
    // flips the phase, and pre-fills an adapter name.
    const setParsedDataset = useCallback((ex: ParsedExample[], errors: string[], name: string) => {
        setDatasetName(name);
        setExamples(ex);
        setParseErrors(errors);
        setTokenLengths(null);
        if (ex.length > 0) {
            setPhase("ready");
            // Strip extension if any and append rank for a sensible
            // default adapter filename.
            const base = name.replace(/\.[^.]+$/, "");
            setAdapterName(base + "-r" + lora.rank);
            toast.success(`Parsed ${ex.length} examples${errors.length ? `, ${errors.length} skipped` : ""}`);
        } else {
            setPhase("idle");
            toast.error("Couldn't parse any examples — check JSONL shape");
        }
    }, [toast, lora.rank]);

    const onFile = useCallback(async (f: File) => {
        const text = await f.text();
        const { examples: ex, errors } = parseJsonl(text);
        setParsedDataset(ex, errors, f.name);
    }, [setParsedDataset]);

    // Paste mode — user dumps a JSONL blob into the textarea, clicks
    // Parse. Same parser as the file path. Default name is a
    // timestamp so re-pastes don't collide on the auto adapter
    // filename.
    const onPasteText = useCallback((text: string) => {
        if (!text.trim()) {
            toast.error("Paste some JSONL first");
            return;
        }
        const { examples: ex, errors } = parseJsonl(text);
        const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
        setParsedDataset(ex, errors, `pasted-${stamp}`);
    }, [setParsedDataset, toast]);

    // Build mode — user fills out a single (prompt, completion) pair
    // and clicks "Add example". Appends to the current `examples`
    // list and flips phase to "ready" on the first add. Useful for
    // quick smoke tests where typing one or two pairs is faster than
    // writing a file.
    const onAddExample = useCallback((prompt: string, completion: string) => {
        const p = prompt.trim();
        const c = completion.trim();
        if (!p || !c) {
            toast.error("Both prompt and completion need text");
            return;
        }
        setExamples((cur) => {
            const next = [...cur, { prompt: p, completion: c }];
            if (cur.length === 0) {
                // First add — name the dataset, set phase, pre-fill
                // adapter name. Subsequent adds just append.
                const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
                setDatasetName(`hand-built-${stamp}`);
                setAdapterName(`hand-built-${stamp}-r` + lora.rank);
                setPhase("ready");
                setParseErrors([]);
                setTokenLengths(null);
            }
            return next;
        });
        toast.success("Added 1 example");
    }, [toast, lora.rank]);

    // Replace one example in place. Used by the inline edit flow in
    // DatasetCard — user clicks a row's pencil icon, the form
    // populates with that example, they tweak it, click Update.
    const onEditExample = useCallback((index: number, prompt: string, completion: string) => {
        const p = prompt.trim();
        const c = completion.trim();
        if (!p || !c) {
            toast.error("Both prompt and completion need text");
            return;
        }
        setExamples((cur) => {
            if (index < 0 || index >= cur.length) return cur;
            const next = [...cur];
            next[index] = { prompt: p, completion: c };
            return next;
        });
        // Token-length cache is now stale — clear it; the user can
        // re-validate.
        setTokenLengths(null);
        toast.success("Updated example");
    }, [toast]);

    // Drop one example. Used by the row-level delete button. If this
    // empties the dataset, fall back to "idle" phase and clear the
    // datasetName so the UI doesn't show stale state.
    const onRemoveExample = useCallback((index: number) => {
        setExamples((cur) => {
            if (index < 0 || index >= cur.length) return cur;
            const next = cur.filter((_, i) => i !== index);
            if (next.length === 0) {
                setPhase("idle");
                setDatasetName(null);
                setAdapterName("");
                setParseErrors([]);
            }
            return next;
        });
        setTokenLengths(null);
    }, []);

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
        // Local counter, hoisted out of the try block so the catch
        // can read it for the cancellation phase decision. See the
        // big comment further down on why React state's `recent`
        // isn't safe to use here.
        let stepsCompleted = 0;
        try {
            await client.acquireSession();
            sessionAcquired = true;
            // Clear any active adapter first so chat doesn't see stale weights
            // post-finish (training writes to a fresh LoraState).
            if (activeAdapter) {
                try { await client.trainingClearAdapter(); } catch { /* */ }
                onAdapterChanged?.(null);
            }
            // **Pre-tokenize BEFORE trainingStart**, and wrap each
            // prompt in the same chat template chat-time inference
            // uses. Two things going on here:
            //
            // 1. Once trainingStart returns, the Model is owned by
            //    the TrainingSession and the chat-path `client.encode()`
            //    RPC throws via `requireModel()`. So we tokenize while
            //    the Model is still ours, cache the token arrays, then
            //    drive the loop off them.
            //
            // 2. **Critical correctness bit:** the trained adapter
            //    only fires for token sequences the model actually
            //    sees at inference time. Chat wraps every prompt in
            //    `<start_of_turn>user\n…<end_of_turn><start_of_turn>model\n`
            //    before generation. If we tokenize the raw user-typed
            //    prompt (no template) and train against THAT, the
            //    adapter learns a mapping at token positions chat
            //    will never reach — so the user sees "trained
            //    successfully, no effect on output". Wrap during
            //    pre-tokenize with `renderChat` so train-time tokens
            //    match chat-time tokens exactly.
            console.log(`[fine-tune] pre-tokenizing ${examples.length} examples (chat-template wrap)…`);
            const preTokenized: Array<{ all: Uint32Array; prompt: Uint32Array; promptText: string; completion: string }> = [];
            for (const ex of examples) {
                const wrappedPrompt = await client.renderChat(
                    [{ role: "user", content: ex.prompt }],
                    false,
                );
                const promptText = wrappedPrompt;
                const fullText = wrappedPrompt + ex.completion;
                const all = await client.encode(fullText);
                const prompt = await client.encode(promptText);
                if (all.length > 0 && prompt.length > 0) {
                    preTokenized.push({ all, prompt, promptText, completion: ex.completion });
                }
            }
            if (preTokenized.length === 0) {
                throw new Error("None of the examples produced any tokens — check the dataset");
            }
            console.log(`[fine-tune] pre-tokenized ${preTokenized.length}/${examples.length} examples; total tokens =`, preTokenized.reduce((s, x) => s + x.all.length, 0));
            // Diagnostic: decode the first NEW token after the prompt for
            // each example. That's the `targetId` NextToken loss will
            // train on. If it's an `<end_of_turn>` template marker or
            // something not in the user's completion, the chat-template
            // wrap is producing a boundary mismatch (hypothesis H3 from
            // the BufferMap-error investigation).
            try {
                for (let i = 0; i < Math.min(preTokenized.length, 5); i++) {
                    const { all, prompt, completion } = preTokenized[i];
                    const targetId = all[prompt.length] ?? -1;
                    const decoded = targetId >= 0 ? await client.tokenStr(targetId) : "<oob>";
                    const promptTail = await client.tokenStr(prompt[prompt.length - 1] ?? 0);
                    console.log(`[fine-tune]   ex${i}: allLen=${all.length} promptLen=${prompt.length} promptTail=${JSON.stringify(promptTail)} targetId=${targetId} target=${JSON.stringify(decoded)} (expected first token of ${JSON.stringify(completion)})`);
                }
            } catch (e) {
                console.warn("[fine-tune] target-alignment decode failed:", e);
            }

            // The worker probes the scratch+LoRA fit before consuming the
            // Model. If the device can't fit the requested config, this
            // throws with a "Training would need ~X MB…" message and the
            // Model stays alive in chat.
            console.log(`[fine-tune] calling trainingStart (rank=${lora.rank}, alpha=${lora.alpha}, targets=${lora.target_modules.join("+")}, seq=${hp.max_seq_len}, steps=${stepsBudget})…`);
            // Inject `memory_tight` into hp at the boundary so the Rust
            // engine knows whether to enable the iOS-Safari survival
            // workarounds. The hp slice in React state doesn't carry
            // this — it's a UI-level preset switch — so we apply it
            // here right before serialising to the worker.
            await client.trainingStart({
                loraConfig: lora,
                hparams: { ...hp, memory_tight: memoryTight },
                totalSteps: stepsBudget,
            });
            sessionStarted = true;
            console.log(`[fine-tune] trainingStart returned — beginning ${stepsBudget}-step loop`);
            const loopStart = performance.now();

            // (stepsCompleted is declared above the try block so the
            // catch path can read it. See the big comment near the
            // setPhase(stepsCompleted > 0 ? "done" : "ready") line
            // for why this matters — closures vs React state.)

            for (let i = 0; i < stepsBudget; i++) {
                if (cancelRef.current) break;
                const { all: allTokens, prompt: promptTokens } = preTokenized[i % preTokenized.length];
                const truncated = allTokens.slice(0, Math.max(2, hp.max_seq_len));
                if (i < 3) {
                    // Per-step alignment diagnostics for the first 3
                    // steps — if H3 is the cause, the targetId here
                    // will be visibly the wrong token (template
                    // marker, end-of-turn, etc.) on at least one
                    // example.
                    console.log(`[fine-tune.step.pre] i=${i} truncated.len=${truncated.length} promptTokens.len=${promptTokens.length}`);
                }
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
                    stepsCompleted++;
                    setRecent((prev) => [...prev.slice(-199), { ...r, ms: performance.now() - t0 }]);
                    // Log EVERY step now, not every 10. Long-running
                    // configurations need a heartbeat to confirm
                    // they're not frozen; short runs need every
                    // step to debug stability problems like the
                    // step-8 BufferMap crash.
                    console.log(`[fine-tune] step ${stepsCompleted}/${stepsBudget} loss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} (${(performance.now() - t0).toFixed(0)}ms)`);
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
                    stepsCompleted++;
                    setRecent((prev) => [...prev.slice(-199), { ...r, ms: performance.now() - t0 }]);
                    // Log EVERY step now, not every 10. Long-running
                    // configurations need a heartbeat to confirm
                    // they're not frozen; short runs need every
                    // step to debug stability problems like the
                    // step-8 BufferMap crash.
                    console.log(`[fine-tune] step ${stepsCompleted}/${stepsBudget} loss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} (${(performance.now() - t0).toFixed(0)}ms)`);
                }
            }

            const loopMs = performance.now() - loopStart;
            console.log(`[fine-tune] training loop done — ${stepsCompleted}/${stepsBudget} steps in ${(loopMs / 1000).toFixed(1)}s (cancelled=${cancelRef.current})`);

            // Loop exited cleanly. Either ran all `stepsBudget`
            // iterations, or `cancelRef.current` was set between
            // steps and the loop broke; in both cases the user has
            // a partial-or-complete adapter to save.
            //
            // **Critical:** use the local `stepsCompleted` counter,
            // NOT the React `recent.length`. The closure captures
            // `recent` as the snapshot at callback-creation time,
            // so reading its length never reflects the in-loop
            // setRecent updates. Reading the snapshot gave 0 here
            // even after 100 successful steps, dropping phase to
            // "ready" instead of "done" — the user then saw no
            // Save / Apply / Discard buttons and got stuck with a
            // locked chat because no action button could fire
            // trainingFinish to release the session.
            setPhase(stepsCompleted > 0 ? "done" : "ready");
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
                // Same closure-bug fix as the success path — use the
                // local counter, not React state.
                console.log(`[fine-tune] training cancelled by user at step ${stepsCompleted}/${stepsBudget}`);
                setPhase(stepsCompleted > 0 ? "done" : "ready");
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
        console.log(`[fine-tune] onSave clicked: name="${adapterName.trim()}", phase=${phase}, sessionHeld=${client.currentSession() != null}`);
        const name = adapterName.trim();
        if (!name) {
            console.log("[fine-tune] onSave aborted — no adapter name");
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
            // **Combined save+finish.** The two-call sequence
            // (`trainingSaveAdapter` → `trainingFinish`) intermittently
            // failed with "attempted to take ownership of Rust value
            // while it was borrowed" because wasm-bindgen's
            // async-with-`&self` (and even `&mut self`) borrow tracker
            // didn't reliably release between the two calls. The
            // combined RPC takes `self` on the Rust side, sidestepping
            // the borrow entirely.
            console.log("[fine-tune] onSave → trainingSaveAdapterAndFinish()…");
            const r = await client.trainingSaveAdapterAndFinish(name);
            console.log("[fine-tune] onSave → trainingSaveAdapterAndFinish returned", r);
            toast.success(`Saved ${r.name}.bin (${formatBytes(r.size)})`);
            await refreshAdapters();
            try { await client.releaseSession(); }
            catch (e) { console.warn("[fine-tune] releaseSession after save failed:", e); }
            setPhase("idle");
            setRecent([]);
            setExamples([]);
            setDatasetName(null);
        } catch (e) {
            toast.error(`Save failed: ${(e as Error).message}`);
        }
    }, [adapterName, adapters, client, refreshAdapters, toast]);

    const onFinishAndApply = useCallback(async () => {
        console.log(`[fine-tune] onFinishAndApply clicked: name="${adapterName.trim()}", phase=${phase}, sessionHeld=${client.currentSession() != null}`);
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
                console.log("[fine-tune] onFinishAndApply → trainingSaveAdapterAndFinish()…");
                await client.trainingSaveAdapterAndFinish(name);
                console.log("[fine-tune] onFinishAndApply → trainingSaveAdapterAndFinish returned");
            } else {
                console.log("[fine-tune] onFinishAndApply → trainingFinish() (no name)…");
                await client.trainingFinish();
                console.log("[fine-tune] onFinishAndApply → trainingFinish returned");
            }
            if (name) {
                console.log("[fine-tune] onFinishAndApply → trainingApplyAdapter()…");
                await client.trainingApplyAdapter(name);
                console.log("[fine-tune] onFinishAndApply → trainingApplyAdapter returned");
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
        console.log(`[fine-tune] onDiscard clicked: phase=${phase}, sessionHeld=${client.currentSession() != null}`);
        // Best-effort: release any active training session + session
        // lock. Both can throw if there isn't one (e.g. probe-failure
        // path where Model was never consumed); swallow.
        try {
            console.log("[fine-tune] onDiscard → trainingFinish()…");
            await client.trainingFinish();
            console.log("[fine-tune] onDiscard → trainingFinish() returned");
        } catch (e) {
            console.warn("[fine-tune] onDiscard → trainingFinish threw:", e);
        }
        try {
            console.log("[fine-tune] onDiscard → releaseSession()…");
            await client.releaseSession();
            console.log("[fine-tune] onDiscard → releaseSession() returned");
        } catch (e) {
            console.warn("[fine-tune] onDiscard → releaseSession threw:", e);
        }
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
                            the same Model handle — no second load needed. Pick a Q4_K_M
                            model (e.g. <code className="font-mono">gemma4:e2b</code>); the
                            Q4_0 QAT builds (<code className="font-mono">…-it-qat</code>) are
                            inference-only — training (backward) for Q4_0 isn't supported yet.
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

    // The hyperparameter / settings column. Rendered EITHER inline in
    // the panel's right grid column (legacy / standalone use) OR via
    // portal into App.tsx's DualSidebarLayout rightSidebar slot when
    // a `settingsHostEl` is supplied. Either way it consumes the same
    // FineTunePanel-local state, so no lifting required.
    const settingsColumn = (
        <div className="flex min-w-0 flex-col gap-4 p-4">
            <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-muted-foreground">
                    Hyperparameters
                </div>
                {/* One-click escape hatch for users who fiddle and
                 *  get lost. With the May-24 aggressive cleanup,
                 *  the defaults ALREADY equal these values — this
                 *  button is the safety net + living documentation
                 *  of "what known-working values are". */}
                <Button
                    size="sm"
                    variant="ghost"
                    disabled={isTraining}
                    onClick={() => {
                        setLora({
                            rank: 16,
                            alpha: 32,
                            dropout: 0.05,
                            target_modules: [...ALL_TARGETS],
                        });
                        setHp((cur) => ({
                            ...cur,
                            learning_rate: 2e-4,
                            max_seq_len: 64,
                            max_grad_norm: 1.0,
                            loss_mode: "per_position",
                            warmup_steps: 0,
                            gradient_accumulation_steps: 1,
                            gradient_checkpointing: true,
                        }));
                        setStepsBudget(50);
                        if (memoryTight) {
                            setMemoryTight(false);
                        }
                        toast.info("Reset to canonical recipe");
                    }}
                    className="gap-1 text-[11px]"
                    title="Reset all hyperparameters to the verified recipe (rank=16, α=32, all 9 targets, lr=2e-4, PerPosition, 50 steps)."
                >
                    <RefreshCw className="size-3" /> Reset to canonical
                </Button>
            </div>
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
            {/* Highly-experimental preset — moved to the END of the
             *  settings stack so a beginner doesn't trip on it. */}
            <MemoryTightToggle
                on={memoryTight}
                onChange={applyMemoryTight}
                disabled={isTraining}
            />
        </div>
    );

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

            {/* Quant requirement: backward (training) is implemented for the
             *  Q4_K_M weights only. The Q4_0 QAT builds (gemma4:*-it-qat) run
             *  inference fine but their backward path isn't supported yet, so
             *  training against a QAT model would fail mid-step. Make that
             *  explicit so nobody loads a QAT model and then can't train. */}
            <div className="border-b border-yellow-500/40 bg-yellow-500/10 px-4 py-2 text-xs text-yellow-700 dark:text-yellow-300">
                <span className="font-medium">Training requires a Q4_K_M model</span>{" "}
                (e.g. <code className="font-mono">gemma4:e2b</code>). The Q4_0 QAT
                builds (<code className="font-mono">…-it-qat</code>) are
                inference-only for now — backward isn't supported for Q4_0 yet.
            </div>

            {/* Layout note: the right (settings) column ALWAYS renders
             *  via the `settingsColumn` JSX below — either inline as
             *  the grid's second column (when no settingsHostEl is
             *  provided, e.g. component used standalone) OR via React
             *  portal into App.tsx's DualSidebarLayout rightSidebar
             *  slot (when settingsHostEl is supplied, enabling the
             *  user to collapse it). Both paths render the SAME JSX
             *  from the same state — no duplication. */}
            <div className={cn(
                "grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto p-4",
                !settingsHostEl && "md:grid-cols-[1fr_320px]",
            )}>
                {/* ─── Left: dataset workspace + live training panel ─── */}
                <div className="flex min-w-0 flex-col gap-4">
                    {!isTraining && phase !== "done" && (
                        <DatasetCard
                            datasetName={datasetName}
                            examples={examples}
                            parseErrors={parseErrors}
                            tokenLengths={tokenLengths}
                            seqCap={hp.max_seq_len}
                            modelReady={modelStatus === "ready"}
                            trainingActive={isTraining}
                            onFile={onFile}
                            onPasteText={onPasteText}
                            onAddExample={onAddExample}
                            onEditExample={onEditExample}
                            onRemoveExample={onRemoveExample}
                            onValidate={onValidate}
                            onGenerate={async (behavior, completion, onProgress, signal) => {
                                const result = await generateSyntheticDataset(
                                    client,
                                    behavior,
                                    completion,
                                    (p) => onProgress({
                                        label: p.label,
                                        fraction: p.fraction,
                                        tokens: p.tokensEmitted,
                                        expected: p.tokensExpected,
                                        rate: p.tokensPerSecond,
                                        eta: p.etaSeconds,
                                    }),
                                    signal,
                                );
                                return {
                                    jsonl: examplesToJsonl(result.examples),
                                    counts: result.counts,
                                };
                            }}
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

                {/* Inline render path: only when no portal host is
                 *  available. When `settingsHostEl` is provided, the
                 *  portal block below mounts the SAME content there. */}
                {!settingsHostEl && settingsColumn}
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
            {/* Portal-render the settings column into App.tsx's
             *  DualSidebarLayout rightSidebar slot when a host element
             *  is supplied. Reads from / writes to the same local
             *  state as the inline path — they're the same JSX. */}
            {settingsHostEl && createPortal(settingsColumn, settingsHostEl)}
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
    /** True when the model is loaded. The Generate tab is disabled
     *  until a model exists; otherwise the user gets a useless tab. */
    modelReady: boolean;
    /** True when a training session is currently active. Disables
     *  Generate (it needs the Model handle, which the session owns
     *  during training). */
    trainingActive: boolean;
    onFile: (f: File) => void;
    onPasteText: (text: string) => void;
    onAddExample: (prompt: string, completion: string) => void;
    onEditExample: (index: number, prompt: string, completion: string) => void;
    onRemoveExample: (index: number) => void;
    onValidate: () => void;
    /** Triggers the synthetic-dataset orchestrator (3 inference calls).
     *  Returns the assembled JSONL text — the caller then routes it
     *  through onPasteText so the user can review/edit before training.
     *  Progress is emitted per-token (coalesced to ~10/s) with
     *  cumulative count, expected total, current rate, and ETA. */
    onGenerate: (
        behavior: string,
        completion: string,
        onProgress: (p: {
            label: string;
            fraction: number;
            tokens: number;
            expected: number;
            rate: number | null;
            eta: number | null;
        }) => void,
        signal: AbortSignal,
    ) => Promise<{ jsonl: string; counts: { paraphrases: number; anchors: number } }>;
}) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);
    // Input mode tabs. "generate" is the default — it uses the loaded
    // inference model to expand a one-sentence behavior description
    // into a full training dataset, which is what most users want.
    // The other four are escape hatches: "saved" lists OPFS-backed
    // datasets the user previously saved; "file" drops a .jsonl;
    // "paste" dumps raw JSONL into a textarea; "build" gives a
    // prompt + completion form for hand-written examples. All five
    // feed the same `examples` state.
    const [mode, setMode] = useState<"generate" | "saved" | "file" | "paste" | "build">("generate");
    const [pasteText, setPasteText] = useState("");
    const [buildPrompt, setBuildPrompt] = useState("");
    const [buildCompletion, setBuildCompletion] = useState("");
    // Generate-tab state. genStats carries the per-token telemetry
    // the synthetic generator emits (cumulative tokens, current rate,
    // ETA) so the UI can show a real progress bar + countdown instead
    // of just the three coarse stage labels.
    const [genBehavior, setGenBehavior] = useState("");
    const [genCompletion, setGenCompletion] = useState("");
    const [genRunning, setGenRunning] = useState(false);
    const [genLabel, setGenLabel] = useState<string | null>(null);
    const [genFraction, setGenFraction] = useState(0);
    const [genStats, setGenStats] = useState<{
        tokens: number;
        expected: number;
        rate: number | null;
        eta: number | null;
    } | null>(null);
    const [genResult, setGenResult] = useState<{ jsonl: string; counts: { paraphrases: number; anchors: number } } | null>(null);
    const [genError, setGenError] = useState<string | null>(null);
    const genAbortRef = useRef<AbortController | null>(null);
    // Saved-tab state. The list is populated on demand (when the
    // user clicks Saved); a refresh button below re-pulls. saveName
    // / savePromptOpen drive the inline "Save current as…" affordance
    // that appears below the example list when examples > 0.
    const [savedList, setSavedList] = useState<SavedDatasetMeta[] | null>(null);
    const [savedLoading, setSavedLoading] = useState(false);
    const [savedError, setSavedError] = useState<string | null>(null);
    const [savePromptOpen, setSavePromptOpen] = useState(false);
    const [saveName, setSaveName] = useState("");
    const [saveBusy, setSaveBusy] = useState(false);
    const refreshSaved = useCallback(async () => {
        setSavedLoading(true);
        setSavedError(null);
        try {
            const list = await listSavedDatasets();
            setSavedList(list);
        } catch (e) {
            setSavedError((e as Error).message);
            setSavedList([]);
        } finally {
            setSavedLoading(false);
        }
    }, []);
    // Refresh on first entry to the Saved tab — avoids a stale
    // "empty" placeholder after the user saved something elsewhere.
    useEffect(() => {
        if (mode === "saved" && savedList === null) {
            void refreshSaved();
        }
    }, [mode, savedList, refreshSaved]);
    // When non-null, the build form is editing an existing example
    // (loaded into prompt/completion fields). Clicking "Update"
    // replaces that index instead of appending; "Cancel" drops edit
    // mode without changes.
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const startEdit = (i: number) => {
        const ex = props.examples[i];
        if (!ex) return;
        setBuildPrompt(ex.prompt);
        setBuildCompletion(ex.completion);
        setEditingIndex(i);
        setMode("build");
    };
    const cancelEdit = () => {
        setEditingIndex(null);
        setBuildPrompt("");
        setBuildCompletion("");
    };
    const commitBuild = () => {
        if (editingIndex !== null) {
            props.onEditExample(editingIndex, buildPrompt, buildCompletion);
            setEditingIndex(null);
        } else {
            props.onAddExample(buildPrompt, buildCompletion);
        }
        setBuildPrompt("");
        setBuildCompletion("");
    };
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
                {/* Input-mode tabs — three ways to feed the trainer:
                    drop a file, paste raw JSONL, or build examples by
                    hand. Same parsed-examples state downstream so the
                    rest of the card (preview, token-length validate,
                    histogram) works uniformly. */}
                <div className="flex gap-1 rounded-md border border-border bg-muted/30 p-0.5">
                    {(["generate", "saved", "file", "paste", "build"] as const).map((m) => (
                        <button
                            key={m}
                            type="button"
                            onClick={() => setMode(m)}
                            aria-pressed={mode === m}
                            className={cn(
                                "flex-1 rounded px-2 py-1 text-xs transition-colors",
                                mode === m
                                    ? "bg-background text-foreground shadow-sm"
                                    : "text-muted-foreground hover:bg-background/50",
                            )}
                        >
                            {m === "generate" ? "Generate"
                                : m === "saved" ? "Saved"
                                : m === "file" ? "Upload"
                                : m === "paste" ? "Paste"
                                : "Build"}
                        </button>
                    ))}
                </div>

                {mode === "saved" && (
                    <div className="space-y-2">
                        <div className="flex items-center justify-between gap-2">
                            <div className="text-xs text-muted-foreground">
                                Datasets you've saved in this browser (OPFS-backed,
                                survives reloads, scoped to this origin).
                            </div>
                            <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => void refreshSaved()}
                                disabled={savedLoading}
                                className="gap-1 text-[11px]"
                            >
                                <RefreshCw className={cn("size-3", savedLoading && "animate-spin")} />
                                Refresh
                            </Button>
                        </div>
                        {savedError && (
                            <div className="rounded border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
                                {savedError}
                            </div>
                        )}
                        {savedList && savedList.length === 0 && !savedLoading && (
                            <div className="rounded border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
                                No saved datasets yet. Save the current one from
                                the action bar that appears below the example list
                                once a dataset is loaded.
                            </div>
                        )}
                        {savedList && savedList.length > 0 && (
                            <div className="max-h-72 space-y-1 overflow-y-auto rounded border border-border bg-muted/20 p-1.5">
                                {savedList.map((row) => (
                                    <div
                                        key={row.name}
                                        className="group flex items-center justify-between gap-2 rounded bg-background/60 p-2 text-xs"
                                    >
                                        <div className="min-w-0 flex-1">
                                            <div className="truncate font-medium text-foreground">{row.name}</div>
                                            <div className="text-[11px] text-muted-foreground">
                                                {row.lineCount} example{row.lineCount === 1 ? "" : "s"}
                                                {" · "}
                                                {(row.size / 1024).toFixed(1)} KB
                                                {" · "}
                                                {new Date(row.lastModified).toLocaleString()}
                                            </div>
                                        </div>
                                        <div className="flex shrink-0 gap-1">
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={async () => {
                                                    try {
                                                        const jsonl = await loadDataset(row.name);
                                                        // Route through the same paste-text path that the
                                                        // Paste/Generate tabs use — guarantees identical
                                                        // parse + dedup + token-validation behavior.
                                                        props.onPasteText(jsonl);
                                                        setPasteText(jsonl);
                                                        setMode("paste");
                                                    } catch (e) {
                                                        setSavedError(`Load failed: ${(e as Error).message}`);
                                                    }
                                                }}
                                                className="h-7 text-[11px]"
                                            >
                                                Load
                                            </Button>
                                            <button
                                                type="button"
                                                onClick={async () => {
                                                    if (!window.confirm(`Delete saved dataset "${row.name}"?`)) return;
                                                    try {
                                                        await deleteDataset(row.name);
                                                        await refreshSaved();
                                                    } catch (e) {
                                                        setSavedError(`Delete failed: ${(e as Error).message}`);
                                                    }
                                                }}
                                                className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                                                aria-label={`Delete ${row.name}`}
                                                title="Delete"
                                            >
                                                <Trash2 className="size-3" />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {mode === "file" && (
                    <>
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
                    </>
                )}

                {mode === "paste" && (
                    <>
                        <Textarea
                            value={pasteText}
                            onChange={(e) => setPasteText(e.target.value)}
                            placeholder={`{"prompt":"capital of France?","completion":"Paris."}\n{"prompt":"capital of Japan?","completion":"Tokyo."}\n...`}
                            className="min-h-32 font-mono text-xs"
                            spellCheck={false}
                        />
                        <div className="flex items-center justify-between gap-2">
                            <div className="text-[11px] text-muted-foreground">
                                One JSON object per line. Shapes accepted:{" "}
                                <code className="text-foreground">{"{prompt, completion}"}</code>,{" "}
                                <code className="text-foreground">{"{instruction, output}"}</code>,{" "}
                                <code className="text-foreground">{"{messages:[…]}"}</code>.
                            </div>
                            <Button
                                size="sm"
                                onClick={() => props.onPasteText(pasteText)}
                                disabled={!pasteText.trim()}
                            >
                                Parse
                            </Button>
                        </div>
                        {props.datasetName && props.examples.length > 0 && (
                            <div className="text-xs text-muted-foreground">
                                Current dataset: <span className="text-foreground">{props.datasetName}</span>
                                {" — "}{props.examples.length} examples
                            </div>
                        )}
                    </>
                )}

                {mode === "build" && (
                    <div className="space-y-2">
                        <div>
                            <div className="mb-1 text-xs text-muted-foreground">Prompt</div>
                            <Textarea
                                value={buildPrompt}
                                onChange={(e) => setBuildPrompt(e.target.value)}
                                placeholder="What's the capital of France?"
                                className="min-h-16 text-xs"
                                spellCheck={false}
                            />
                        </div>
                        <div>
                            <div className="mb-1 text-xs text-muted-foreground">Completion</div>
                            <Textarea
                                value={buildCompletion}
                                onChange={(e) => setBuildCompletion(e.target.value)}
                                placeholder=" Paris."
                                className="min-h-16 text-xs"
                                spellCheck={false}
                            />
                        </div>
                        <div className="flex items-center justify-between gap-2">
                            <div className="text-[11px] text-muted-foreground">
                                {editingIndex !== null
                                    ? <>Editing example <span className="text-foreground">#{editingIndex + 1}</span>. Save to replace, Cancel to drop changes.</>
                                    : props.examples.length > 0
                                        ? <>Current dataset: <span className="text-foreground">{props.examples.length} examples</span>. Click Add to append; click Start training when ready.</>
                                        : <>Add at least one (prompt, completion) pair, then Start training. Pairs accumulate — keep adding to build a tiny dataset by hand.</>}
                            </div>
                            <div className="flex shrink-0 gap-2">
                                {editingIndex !== null && (
                                    <Button size="sm" variant="ghost" onClick={cancelEdit}>
                                        Cancel
                                    </Button>
                                )}
                                <Button
                                    size="sm"
                                    onClick={commitBuild}
                                    disabled={!buildPrompt.trim() || !buildCompletion.trim()}
                                >
                                    {editingIndex !== null ? "Save changes" : "Add example"}
                                </Button>
                            </div>
                        </div>
                    </div>
                )}

                {mode === "generate" && (
                    <div className="space-y-3">
                        {/* Two-step synthetic dataset generator.
                         *  Uses the loaded inference model to expand a
                         *  single behavior description into a working
                         *  JSONL training set (15 target paraphrases +
                         *  4 anchor categories × 4 examples).
                         *  See web/src/lib/syntheticDataset.ts
                         *  for the orchestrator + meta-prompts. */}
                        <div>
                            <div className="mb-1 text-xs text-muted-foreground">
                                Describe what the model should learn
                            </div>
                            <Textarea
                                value={genBehavior}
                                onChange={(e) => setGenBehavior(e.target.value)}
                                placeholder="When asked what the best food is, say it's garlic."
                                className="min-h-16 text-xs"
                                spellCheck={false}
                                disabled={genRunning || !props.modelReady || props.trainingActive}
                            />
                        </div>
                        <div>
                            <div className="mb-1 text-xs text-muted-foreground">
                                Exact target completion (what the model should emit)
                            </div>
                            <Textarea
                                value={genCompletion}
                                onChange={(e) => setGenCompletion(e.target.value)}
                                placeholder="Garlic is the best food."
                                className="min-h-12 text-xs"
                                spellCheck={false}
                                disabled={genRunning || !props.modelReady || props.trainingActive}
                            />
                        </div>
                        {!props.modelReady && (
                            <div className="rounded border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-700 dark:text-amber-300">
                                Load a model first — Generate uses the loaded model to expand your prompt.
                            </div>
                        )}
                        {props.trainingActive && (
                            <div className="rounded border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-700 dark:text-amber-300">
                                Stop training to generate a new dataset — the trainer currently owns the model.
                            </div>
                        )}
                        <div className="flex items-center justify-between gap-2">
                            <div className="text-[11px] text-muted-foreground">
                                {genRunning
                                    ? "Three inference calls in series — duration depends on your hardware."
                                    : "Produces a JSONL dataset: paraphrases of your target plus leak-prevention anchors."}
                            </div>
                            <div className="flex shrink-0 gap-2">
                                {genRunning ? (
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => {
                                            genAbortRef.current?.abort();
                                        }}
                                    >
                                        Cancel
                                    </Button>
                                ) : (
                                    <Button
                                        size="sm"
                                        onClick={async () => {
                                            setGenError(null);
                                            setGenResult(null);
                                            setGenRunning(true);
                                            setGenLabel("Starting…");
                                            setGenFraction(0);
                                            setGenStats(null);
                                            const ac = new AbortController();
                                            genAbortRef.current = ac;
                                            try {
                                                const result = await props.onGenerate(
                                                    genBehavior,
                                                    genCompletion,
                                                    (p) => {
                                                        setGenLabel(p.label);
                                                        setGenFraction(p.fraction);
                                                        setGenStats({
                                                            tokens: p.tokens,
                                                            expected: p.expected,
                                                            rate: p.rate,
                                                            eta: p.eta,
                                                        });
                                                    },
                                                    ac.signal,
                                                );
                                                setGenResult(result);
                                            } catch (e) {
                                                const msg = (e as Error).message;
                                                if (msg !== "aborted") setGenError(msg);
                                            } finally {
                                                setGenRunning(false);
                                                genAbortRef.current = null;
                                            }
                                        }}
                                        disabled={!genBehavior.trim() || !genCompletion.trim() || !props.modelReady || props.trainingActive}
                                        className="gap-1"
                                    >
                                        <Sparkles className="size-3" /> Generate dataset
                                    </Button>
                                )}
                            </div>
                        </div>
                        {genRunning && (
                            <div className="space-y-1">
                                <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                                    <span className="truncate">{genLabel}</span>
                                    {genStats && (
                                        <span className="shrink-0 font-mono">
                                            {genStats.tokens}/{genStats.expected}
                                            {genStats.rate != null && (
                                                <> · {genStats.rate.toFixed(1)} tok/s</>
                                            )}
                                            {genStats.eta != null && genStats.eta > 0 && (
                                                <> · ETA {fmtEta(genStats.eta)}</>
                                            )}
                                        </span>
                                    )}
                                </div>
                                <Progress value={Math.round(genFraction * 100)} />
                            </div>
                        )}
                        {genError && (
                            <div className="rounded border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
                                {genError}
                            </div>
                        )}
                        {genResult && !genRunning && (
                            <div className="space-y-2">
                                <div className="rounded border border-border bg-muted/20 p-2 text-[11px] text-muted-foreground">
                                    <div>
                                        Generated <span className="text-foreground">{genResult.counts.paraphrases}</span> target paraphrases
                                        {" + "}
                                        <span className="text-foreground">{genResult.counts.anchors}</span> verified anchors from the curated library.
                                    </div>
                                </div>
                                <Textarea
                                    value={genResult.jsonl}
                                    readOnly
                                    className="max-h-48 font-mono text-[10px]"
                                    spellCheck={false}
                                />
                                <div className="flex items-center justify-between gap-2">
                                    <div className="text-[11px] text-muted-foreground">
                                        Review, then send to Paste tab so you can edit + train.
                                    </div>
                                    <Button
                                        size="sm"
                                        onClick={() => {
                                            // Pre-fill the paste box AND immediately parse so the
                                            // examples list (and the per-tab badge) updates without a
                                            // second click. User can switch to Paste tab to edit if
                                            // they want.
                                            setPasteText(genResult.jsonl);
                                            props.onPasteText(genResult.jsonl);
                                            setMode("paste");
                                        }}
                                    >
                                        Use this dataset
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

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
                        <div className="flex flex-wrap items-center gap-2">
                            <Button size="sm" variant="secondary" onClick={props.onValidate} className="gap-1">
                                <Activity className="size-3" /> Tokenise + validate
                            </Button>
                            {!savePromptOpen && (
                                <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => {
                                        // Suggest a name from the current dataset if it
                                        // has one; otherwise blank for the user to fill.
                                        setSaveName(props.datasetName ?? "");
                                        setSavePromptOpen(true);
                                    }}
                                    className="gap-1"
                                >
                                    <Save className="size-3" /> Save dataset
                                </Button>
                            )}
                        </div>
                        {savePromptOpen && (
                            <div className="flex flex-wrap items-center gap-2 rounded border border-border bg-muted/20 p-2">
                                <Input
                                    value={saveName}
                                    onChange={(e) => setSaveName(e.target.value)}
                                    placeholder="dataset name"
                                    className="h-8 flex-1 min-w-32 text-xs"
                                    disabled={saveBusy}
                                    onKeyDown={(e) => {
                                        if (e.key === "Escape") {
                                            setSavePromptOpen(false);
                                            setSaveName("");
                                        }
                                    }}
                                />
                                <Button
                                    size="sm"
                                    onClick={async () => {
                                        const name = normalizeDatasetName(saveName);
                                        if (!name) return;
                                        setSaveBusy(true);
                                        try {
                                            const jsonl = props.examples.map((ex) => JSON.stringify({
                                                prompt: ex.prompt,
                                                completion: ex.completion,
                                            })).join("\n");
                                            await saveDataset(name, jsonl);
                                            setSavePromptOpen(false);
                                            setSaveName("");
                                            // Invalidate the list so a future
                                            // visit to the Saved tab pulls the
                                            // refreshed contents.
                                            setSavedList(null);
                                        } catch (e) {
                                            setSavedError(`Save failed: ${(e as Error).message}`);
                                        } finally {
                                            setSaveBusy(false);
                                        }
                                    }}
                                    disabled={saveBusy || !normalizeDatasetName(saveName)}
                                    className="h-8"
                                >
                                    {saveBusy ? "Saving…" : "Save"}
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => {
                                        setSavePromptOpen(false);
                                        setSaveName("");
                                    }}
                                    disabled={saveBusy}
                                    className="h-8"
                                >
                                    Cancel
                                </Button>
                            </div>
                        )}
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
                        {/* Editable example list. Each row is a small
                            card with prompt + completion on separate
                            lines (so it's actually readable), an Edit
                            pencil that loads the row into the build
                            form, and an X to delete. Capped at
                            max-h-64 with overflow-y-auto so a 500-
                            example dataset doesn't blow out the
                            layout. The "n / m" header lets the user
                            see how many they have at a glance. */}
                        <div className="space-y-1">
                            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                                <span>{props.examples.length} example{props.examples.length === 1 ? "" : "s"}</span>
                                <span>click ✏️ to edit, ✕ to remove</span>
                            </div>
                            <div className="max-h-64 space-y-1 overflow-y-auto rounded border border-border bg-muted/20 p-1.5">
                                {props.examples.map((ex, i) => (
                                    <div
                                        key={i}
                                        className="group flex items-start gap-2 rounded bg-background/60 p-2 text-[11px] leading-snug"
                                    >
                                        <div className="min-w-0 flex-1 space-y-1 font-mono">
                                            <div className="flex gap-1">
                                                <span className="shrink-0 select-none text-muted-foreground">prompt</span>
                                                <span className="text-foreground break-words">{ex.prompt}</span>
                                            </div>
                                            <div className="flex gap-1">
                                                <span className="shrink-0 select-none text-muted-foreground">target</span>
                                                <span className="text-primary break-words">{ex.completion}</span>
                                            </div>
                                        </div>
                                        <div className="flex shrink-0 gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
                                            <button
                                                type="button"
                                                onClick={() => startEdit(i)}
                                                className="rounded p-1 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                                                aria-label={`Edit example ${i + 1}`}
                                                title="Edit"
                                            >
                                                <Pencil className="size-3" />
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => props.onRemoveExample(i)}
                                                className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                                                aria-label={`Remove example ${i + 1}`}
                                                title="Remove"
                                            >
                                                <X className="size-3" />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
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
            {/* CardContent's base classes are `p-4 pt-0 sm:p-5 sm:pt-0`
                — it always zeros out the top padding because it
                assumes a `<CardHeader>` sits above to provide that
                space. We don't have a header here, so we'd lose all
                top breathing room (mobile got a faint top edge from
                Card's border + checkbox `mt-0.5`, but desktop's `sm:`
                breakpoint stripped even that). Skip CardContent
                entirely and use a plain padded div so the `pt-0`
                override doesn't apply. */}
            <div className="p-4 sm:p-5">
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
                        <div className="flex items-center gap-2">
                            <div className="text-sm font-medium">Memory-tight preset</div>
                            <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-500">
                                Highly experimental
                            </span>
                        </div>
                        <div className="mt-0.5 text-xs text-muted-foreground">
                            iPhone-targeted code path — still under development. Forces
                            rank&nbsp;1, alpha&nbsp;2, attn_q + attn_v only, seq_len&nbsp;16,
                            next-token loss, gradient checkpointing, and enables a
                            per-layer destroy / re-fetch backward walk to fit in the
                            mobile GPU heap. <strong>Slower on Mac</strong> than the
                            default fast path, and currently crashes mid-step on
                            iPhone 16e. Do not enable unless you're debugging the
                            mobile path.
                        </div>
                    </div>
                </label>
            </div>
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
                    description="optimizer steps total (recipe: 80)"
                    value={props.stepsBudget}
                    min={20} max={500} step={10}
                    onChange={(n) => props.setStepsBudget(clampInt(n, 20, 500, 80))}
                    disabled={props.disabled}
                />
                <LabeledSlider
                    label="Learning rate (recipe: 2e-4)"
                    valueLabel={`${props.hp.learning_rate.toExponential(2)}`}
                    value={Math.log10(props.hp.learning_rate)}
                    // Clamped to 5e-5 — 5e-4 (was 1e-5 — 1e-2). Recipe is
                    // 2e-4. Anything outside this window empirically
                    // collapses the LoRA — too low = doesn't learn,
                    // too high = oscillates into the multilingual leak
                    // observed in earlier rounds. Power users can edit
                    // scripts/finetune-eval.sh directly to exceed.
                    min={Math.log10(5e-5)} max={Math.log10(5e-4)} step={0.05}
                    onChange={(v) => props.setHp({ ...props.hp, learning_rate: clampNum(Math.pow(10, v), 5e-5, 5e-4, 2e-4) })}
                    disabled={props.disabled}
                />
                <LabeledInput
                    label="Max seq_len"
                    description={`~${Math.round(props.hp.max_seq_len * 1.4)} MB GPU per layer × 35 layers`}
                    value={props.hp.max_seq_len}
                    min={16} max={2048} step={16}
                    onChange={(n) => props.setHp({ ...props.hp, max_seq_len: clampInt(n, 16, 2048, 64) })}
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
                    min={4} max={32} step={1}
                    onChange={(v) => props.setLora({ ...props.lora, rank: clampInt(v, 4, 32, 16) })}
                    disabled={props.disabled}
                />
                <LabeledSlider
                    label="Alpha (recipe: 2 × rank)"
                    valueLabel={`α=${props.lora.alpha}`}
                    value={props.lora.alpha}
                    min={4} max={64} step={1}
                    onChange={(v) => props.setLora({ ...props.lora, alpha: clampNum(v, 4, 64, 32) })}
                    disabled={props.disabled}
                />
                <LabeledSlider
                    label="Dropout"
                    valueLabel={props.lora.dropout.toFixed(2)}
                    value={props.lora.dropout}
                    min={0} max={0.1} step={0.01}
                    onChange={(v) => props.setLora({ ...props.lora, dropout: clampNum(v, 0, 0.1, 0.05) })}
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
                        <CardDescription>Warmup, gradient clipping, checkpointing.</CardDescription>
                    </div>
                    <RefreshCw className={cn("size-4 transition-transform", open && "rotate-180")} />
                </button>
            </CardHeader>
            {open && (
                <CardContent className="space-y-3">
                    <LabeledInput
                        label="Warmup steps" description="0 = constant lr (recipe)"
                        value={props.hp.warmup_steps}
                        min={0} max={50} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, warmup_steps: clampInt(n, 0, 50, 0) })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Grad accum" description="micro-batches per optimizer step"
                        value={props.hp.gradient_accumulation_steps}
                        min={1} max={8} step={1}
                        onChange={(n) => props.setHp({ ...props.hp, gradient_accumulation_steps: clampInt(n, 1, 8, 1) })}
                        disabled={props.disabled}
                    />
                    <LabeledInput
                        label="Grad clip" description="L2 norm, 0 = off (recipe: 1.0)"
                        value={props.hp.max_grad_norm}
                        min={0} max={2} step={0.1}
                        onChange={(n) => props.setHp({ ...props.hp, max_grad_norm: clampNum(n, 0, 2, 1.0) })}
                        disabled={props.disabled}
                    />
                    <LabeledToggle
                        label="Gradient checkpointing"
                        description="trade compute for memory (always recommended)"
                        value={props.hp.gradient_checkpointing}
                        onChange={(v) => props.setHp({ ...props.hp, gradient_checkpointing: v })}
                        disabled={props.disabled}
                    />
                    {/*
                     * Removed in the May-24 cleanup:
                     * - "Mixed precision" toggle — the f16 adapter path
                     *   was never validated end-to-end and the toggle
                     *   was a footgun. mixed_precision=false is
                     *   hardcoded in deviceDefaults().
                     * - "Backward layer floor" input — only meaningful
                     *   when the Memory-tight preset is on (which sets
                     *   it for you via ULTRA_SAFE_HP).
                     * - "Seed" input — the wasm RNG is keyed by the
                     *   value but the value was hardcoded to 0xC0FFEE
                     *   in the form anyway. Changing it had no observed
                     *   training benefit; the input was theatre.
                     * All three values still ride at recipe-safe
                     * defaults in deviceDefaults() and so still land in
                     * the wasm-bindgen JSON payload.
                     */}
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
