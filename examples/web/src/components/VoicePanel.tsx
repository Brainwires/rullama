import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, Loader2, Mic, Play, Trash2, Upload } from "lucide-react";

import iconUrl from "../assets/icon-512.png";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { kokoroBlobUrl, styletts2BlobUrl, styletts2Model } from "@/lib/api";
import { getSharedClone } from "@/lib/clone-client";
import { getSharedTts, type TtsClient, type TtsClip } from "@/lib/tts-client";
import { usePersistedState } from "@/lib/persisted";
import { cn } from "@/lib/utils";
import { addVoice, importVoiceFile, listVoices, onVoicesChanged, removeVoice, voiceVec, type SavedVoice } from "@/lib/voice-library";
import { downloadWav, playPcm } from "@/lib/wav";

// The 28 English Kokoro voicepacks baked into the GGUF (af/am = American F/M, bf/bm = British F/M).
const PRESETS = [
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah", "af_sky", "af_nova",
    "af_alloy", "af_jessica", "af_river",
    "am_michael", "am_fenrir", "am_puck", "am_adam", "am_echo", "am_eric", "am_liam", "am_onyx", "am_santa",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
];
// "af_heart" → "Heart · US ♀". af/am = American, bf/bm = British; f = ♀, m = ♂.
function presetLabel(id: string): string {
    const name = id.slice(3).replace(/^\w/, (c) => c.toUpperCase());
    const region = id[0] === "b" ? "UK" : "US";
    const sex = id[1] === "f" ? "♀" : "♂";
    return `${name} · ${region} ${sex}`;
}

/** `settingsHostEl`: when provided (App's right-sidebar slot), the voice picker + import/delete
 *  render there via portal. `clipsHostEl`: App's left-sidebar slot for the generated-clips list.
 *  Either falls back to an inline column when its host is absent.
 *  `onOpenVoiceLearn`: opens the full-screen Voice-learning (clone training) overlay. */
export function VoicePanel(
    { settingsHostEl, clipsHostEl, onOpenVoiceLearn }:
    { settingsHostEl?: HTMLElement | null; clipsHostEl?: HTMLElement | null; onOpenVoiceLearn?: () => void } = {},
) {
    // Clone-engine precision. Phase 1 ships f32 only; the f16 (memory-tight,
    // mobile-default) variant is wired in Phase 2 — the option is present but
    // disabled so the control + persisted choice exist for forward-compat.
    const [cloneVariant, setCloneVariant] = usePersistedState<"f32" | "f16">("voice:cloneVariant", "f32");
    const tts = useRef<TtsClient | null>(null);
    const [err, setErr] = useState<string | null>(null);
    const [text, setText] = useState("Hello, this is text to speech running entirely in your browser.");
    const [sel, setSel] = useState("k:af_heart"); // "k:<preset>" (Kokoro) or "c:<id>" (cloned)
    const [voices, setVoices] = useState<SavedVoice[]>(() => listVoices());
    const [busy, setBusy] = useState(false);
    const [dlPct, setDlPct] = useState(0);
    const [procPct, setProcPct] = useState(0);
    const [procStage, setProcStage] = useState("");
    const [clips, setClips] = useState<TtsClip[]>([]);
    const [active, setActive] = useState<number>(-1);

    useEffect(() => onVoicesChanged(() => setVoices(listVoices())), []);

    const isClone = sel.startsWith("c:");
    const cloneVoice = isClone ? voices.find((v) => v.id === sel.slice(2)) : undefined;

    const generate = useCallback(async () => {
        if (busy || !text.trim()) return;
        setBusy(true);
        setErr(null);
        setDlPct(0);
        setProcPct(0);
        setProcStage("");
        try {
            let pcm: Float32Array;
            let sr: number;
            let label: string;
            if (isClone) {
                if (!cloneVoice) throw new Error("voice not found");
                const cc = await getSharedClone(styletts2BlobUrl(cloneVariant), styletts2Model(cloneVariant).size, cloneVariant, (f) => setDlPct(Math.round(f * 100)));
                pcm = await cc.synthesize(text.trim(), voiceVec(cloneVoice), (f, s) => {
                    setProcPct(Math.round(f * 100));
                    setProcStage(s);
                });
                sr = cc.sampleRate;
                label = cloneVoice.name;
            } else {
                if (!tts.current) tts.current = await getSharedTts(kokoroBlobUrl(), (f) => setDlPct(Math.round(f * 100)));
                const preset = sel.slice(2);
                pcm = await tts.current.synthesize(text.trim(), preset);
                sr = tts.current.sampleRate;
                label = preset;
            }
            const clip: TtsClip = { pcm, sampleRate: sr, text: text.trim(), voice: label, ts: Date.now() };
            setClips((cs) => [clip, ...cs]);
            setActive(0);
            playPcm(pcm, sr);
        } catch (e) {
            setErr(String(e));
        } finally {
            setBusy(false);
        }
    }, [busy, text, isClone, cloneVoice, sel, cloneVariant]);

    const onImport = useCallback(async (f: File | undefined) => {
        if (!f) return;
        setErr(null);
        try {
            const vec = await importVoiceFile(f);
            const name = prompt("Name this imported voice:", f.name.replace(/\.[^.]+$/, "")) ?? f.name;
            const v = addVoice(name, vec);
            setSel(`c:${v.id}`);
        } catch (e) {
            setErr(String(e));
        }
    }, []);

    // Voice options — rendered in the right sidebar (via portal) or inline as a fallback.
    const voiceSettings = (
        <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Voice</div>
            <select
                value={sel}
                onChange={(e) => setSel(e.target.value)}
                className="h-8 w-full rounded-md border border-border bg-background px-2 text-xs"
            >
                <optgroup label="Preset (Kokoro)">
                    {PRESETS.map((v) => (
                        <option key={v} value={`k:${v}`}>{presetLabel(v)}</option>
                    ))}
                </optgroup>
                {voices.length > 0 && (
                    <optgroup label="My cloned voices">
                        {voices.map((v) => (
                            <option key={v.id} value={`c:${v.id}`}>{v.name}</option>
                        ))}
                    </optgroup>
                )}
            </select>
            <div className="flex flex-wrap items-center gap-2">
                <label className={cn("inline-flex h-8 cursor-pointer items-center gap-1 rounded-md border border-border px-2 text-xs hover:bg-muted/50", busy && "pointer-events-none opacity-50")} title="Import a .f32 voice exported from Voice training">
                    <Upload className="size-3.5" /> Import
                    <input type="file" accept=".f32,application/octet-stream" className="hidden" disabled={busy} onChange={(e) => onImport(e.target.files?.[0])} />
                </label>
                {isClone && cloneVoice && (
                    <Button size="sm" variant="ghost" className="text-muted-foreground hover:text-destructive" title="Delete this cloned voice" onClick={() => { removeVoice(cloneVoice.id); setSel("k:af_heart"); }}>
                        <Trash2 className="size-3.5" /> Delete voice
                    </Button>
                )}
            </div>
            <p className="text-[11px] text-muted-foreground">
                {isClone
                    ? "Cloned voices run the StyleTTS2 engine on your GPU (desktop), with style diffusion for natural prosody — first use downloads a 543 MB model (then OPFS-cached)."
                    : "Preset voices run Kokoro on your GPU via WebGPU. Clone your own voice with Voice learning below."}
            </p>

            {/* Clone (StyleTTS2) model block — precision selector + the Voice-learning launcher. */}
            <div className="flex flex-col gap-2 border-t border-border pt-3">
                <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Clone model</div>
                <label className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                    <span>Precision</span>
                    <select
                        value={cloneVariant}
                        onChange={(e) => setCloneVariant(e.target.value as "f32" | "f16")}
                        className="h-7 rounded-md border border-border bg-background px-2 text-xs"
                        title="f32 = full quality (518MB). f16 = ~half the memory (259MB) for memory-tight devices, but a real quality drop — this deep vocoder degrades under f16."
                    >
                        <option value="f32">f32 — full quality</option>
                        <option value="f16">f16 — ½ memory, lower quality</option>
                    </select>
                </label>
                <Button
                    size="sm"
                    variant="outline"
                    className="h-8 justify-start gap-2 text-xs"
                    onClick={() => onOpenVoiceLearn?.()}
                    title="Record your voice and train a clone (StyleTTS2, desktop)"
                >
                    <Mic className="size-3.5" /> Voice learning…
                </Button>
            </div>
        </div>
    );

    // Generated-clips list — rendered in App's left sidebar (via portal) or
    // inline as a fallback.
    const clipsList = (
        <div className="flex h-full min-h-0 flex-col">
            <div className="border-b border-border px-3 py-2 text-xs font-medium text-muted-foreground">
                Generated ({clips.length})
            </div>
            <ul className="flex-1 overflow-y-auto">
                {clips.map((c, i) => (
                    <li
                        key={c.ts}
                        onClick={() => setActive(i)}
                        className={cn(
                            "group cursor-pointer border-b border-border/50 px-3 py-2 text-xs hover:bg-background/60",
                            i === active && "bg-background",
                        )}
                    >
                        <div className="line-clamp-2 text-foreground">{c.text}</div>
                        <div className="mt-1 flex items-center justify-between text-[10px] text-muted-foreground">
                            <span>{(c.pcm.length / c.sampleRate).toFixed(1)}s · {c.voice}</span>
                            <span className="flex gap-1 opacity-0 group-hover:opacity-100">
                                <button title="Play" onClick={(e) => { e.stopPropagation(); playPcm(c.pcm, c.sampleRate); }}>
                                    <Play className="size-3" />
                                </button>
                                <button title="Save WAV" onClick={(e) => { e.stopPropagation(); downloadWav(c.pcm, c.sampleRate, `tts-${c.ts}`); }}>
                                    <Download className="size-3" />
                                </button>
                                <button title="Delete" onClick={(e) => { e.stopPropagation(); setClips((cs) => cs.filter((x) => x.ts !== c.ts)); }}>
                                    <Trash2 className="size-3" />
                                </button>
                            </span>
                        </div>
                    </li>
                ))}
                {clips.length === 0 && <li className="px-3 py-6 text-center text-xs text-muted-foreground">No clips yet</li>}
            </ul>
        </div>
    );

    return (
        <div className="flex h-full min-h-0">
            {/* artifact sidebar — inline fallback when there's no left-sidebar host. */}
            {!clipsHostEl && (
                <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-muted/20 sm:flex">
                    {clipsList}
                </aside>
            )}

            {/* main */}
            <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
                <div className="flex items-center gap-2">
                    <img src={iconUrl} alt="rullama" className="size-8" draggable={false} />
                    <div className="flex flex-col leading-tight">
                        <span className="text-sm font-semibold tracking-tight">rullama</span>
                        <span className="text-[11px] text-muted-foreground">Voice</span>
                    </div>
                </div>

                <Textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type text to speak…"
                    className="min-h-32 resize-none"
                />

                <div className="flex flex-wrap items-center gap-2">
                    <Button size="sm" onClick={generate} disabled={busy || !text.trim()}>
                        {busy ? <><Loader2 className="size-3.5 animate-spin" /> {dlPct > 0 && dlPct < 100 ? `Loading… ${dlPct}%` : "Generating…"}</> : "Generate speech"}
                    </Button>

                    {active >= 0 && clips[active] && (
                        <>
                            <Button size="sm" variant="outline" onClick={() => playPcm(clips[active].pcm, clips[active].sampleRate)}>
                                <Play className="size-3.5" /> Play
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => downloadWav(clips[active].pcm, clips[active].sampleRate, `tts-${clips[active].ts}`)}>
                                <Download className="size-3.5" /> Save WAV
                            </Button>
                        </>
                    )}
                </div>

                {busy && dlPct > 0 && dlPct < 100 && <Progress value={dlPct} className="h-1" />}
                {busy && isClone && procPct > 0 && (
                    <div className="flex items-center justify-between rounded-md border border-border bg-muted/20 px-2 py-1.5 text-[11px] text-muted-foreground">
                        <span className="truncate">{procStage || "synthesizing"}…</span>
                        <span className="tabular-nums">{procPct}%</span>
                    </div>
                )}
                {err && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>}

                {/* Inline fallback when there's no sidebar host (narrow viewport / no DualSidebarLayout). */}
                {!settingsHostEl && <div className="border-t border-border pt-3">{voiceSettings}</div>}
            </div>

            {settingsHostEl && createPortal(voiceSettings, settingsHostEl)}
            {clipsHostEl && createPortal(clipsList, clipsHostEl)}
        </div>
    );
}
