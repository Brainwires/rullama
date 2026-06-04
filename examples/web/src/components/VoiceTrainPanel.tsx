import { useCallback, useEffect, useRef, useState } from "react";
import { AudioLines, Loader2, Mic, Play, Plus, RotateCcw, Save, Square, Trash2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { STYLETTS2_MODEL, styletts2BlobUrl } from "@/lib/api";
import { getSharedClone } from "@/lib/clone-client";
import { cn } from "@/lib/utils";
import { addVoice } from "@/lib/voice-library";
import { clearSession, loadSession, saveSession } from "@/lib/voice-session";
import { decodeToPcm24k, playPcm } from "@/lib/wav";

const MIN_CLIPS = 1;
const SR = 24000;
// A0 calibration: a single clean clip saturates identity by ~15 s; quantity past that barely helps
// ONE vector — but averaging MANY clips (rejecting noisy outliers) is what makes the voice robust.
const CLIP_CAP_SEC = 15; // each clip encoded up to this; then trimmed-mean across clips

/** Phonetically varied prompts — good phoneme coverage for a timbre signature. */
const DEFAULT_SCRIPT = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on a sunny day.",
    "I really enjoy reading books about science and history.",
    "Please call me back as soon as you possibly can.",
    "The weather today is bright, warm, and absolutely beautiful.",
    "We drove through the mountains and stopped to take photographs.",
    "My favorite meal is fresh bread, cheese, and ripe tomatoes.",
    "Thank you so much — that was incredibly kind of you.",
];

interface SessionClip {
    id: string;
    text: string;
    pcm: Float32Array | null;
    durationSec: number;
    peak?: number; // max abs sample — flags clipping (≈1.0) or too-quiet (low) takes
}

/** Peak |sample| of a clip (loop, not spread — clips are large). */
function peakOf(pcm: Float32Array): number {
    let p = 0;
    for (let i = 0; i < pcm.length; i++) {
        const a = Math.abs(pcm[i]);
        if (a > p) p = a;
    }
    return p;
}

const newId = () => crypto.randomUUID();

/** Trim leading/trailing dead air so silence doesn't drag the mean-mel timbre signature. */
function trimSilence(pcm: Float32Array, thresh = 0.012): Float32Array {
    let start = 0;
    let end = pcm.length;
    while (start < end && Math.abs(pcm[start]) < thresh) start++;
    while (end > start && Math.abs(pcm[end - 1]) < thresh) end--;
    const pad = (SR * 0.05) | 0; // keep 50 ms either side
    start = Math.max(0, start - pad);
    end = Math.min(pcm.length, end + pad);
    if (end - start < SR * 0.2) return pcm; // <200 ms of voice — keep original
    return pcm.slice(start, end);
}

/** Cosine over the acoustic (timbre) half — the first 128 dims of the 256-d style vector. */
function timbreCos(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < 128; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

function meanVec(vs: Float32Array[]): Float32Array {
    const m = new Float32Array(256);
    for (const v of vs) for (let i = 0; i < 256; i++) m[i] += v[i];
    for (let i = 0; i < 256; i++) m[i] /= vs.length;
    return m;
}

/**
 * Robust voice = trimmed mean of per-clip style vectors, dropping the clips whose timbre sits
 * farthest from the centroid (noisy / clipped takes). A0 showed one noisy clip alone scores 0.46
 * vs af_heart (below the 0.70 different-speaker floor), so rejecting outliers is the core lever.
 * Returns the aggregated vector + how many clips were dropped.
 */
function robustVoice(styles: Float32Array[]): { vec: Float32Array; dropped: number } {
    if (styles.length <= 2) return { vec: meanVec(styles), dropped: 0 };
    const c = meanVec(styles);
    const ranked = styles
        .map((v) => ({ v, d: 1 - timbreCos(v, c) }))
        .sort((a, b) => a.d - b.d);
    const keep = Math.max(2, Math.ceil(styles.length * 0.8)); // drop the worst ~20%
    return { vec: meanVec(ranked.slice(0, keep).map((r) => r.v)), dropped: styles.length - keep };
}

const fmtDur = (s: number) => `${s.toFixed(1)}s`;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/**
 * Guided voice cloning: record your own voice over a short script, edit the clip list
 * (re-record / play / delete / add / upload), then clone your timbre with StyleTTS2. Each clip
 * is encoded to its own style vector and robustly averaged (noisy takes dropped) — so more
 * CLEAN clips ⇒ a more faithful voice, and one bad take can't poison it.
 */
export function VoiceTrainPanel() {
    const [clips, setClips] = useState<SessionClip[]>(() =>
        DEFAULT_SCRIPT.map((text) => ({ id: newId(), text, pcm: null, durationSec: 0 })),
    );
    const [recordingId, setRecordingId] = useState<string | null>(null);
    const [recSecs, setRecSecs] = useState(0);
    const [phase, setPhase] = useState<"idle" | "loading" | "encoding" | "synth" | "done" | "error">("idle");
    const [dlPct, setDlPct] = useState(0);
    const [procPct, setProcPct] = useState(0);
    const [procStage, setProcStage] = useState("");
    const [logLines, setLogLines] = useState<string[]>([]);
    const [err, setErr] = useState<string | null>(null);
    const [testText, setTestText] = useState("This is my cloned voice, generated entirely on my own device.");
    const [voiceName, setVoiceName] = useState("My voice");
    const [savedMsg, setSavedMsg] = useState<string | null>(null);

    const mrRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const voiceRef = useRef<Float32Array | null>(null);

    // Cleanup any live mic stream / timer on unmount.
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            streamRef.current?.getTracks().forEach((t) => t.stop());
        };
    }, []);

    // Restore a saved session on mount, then persist (debounced) on every change — so
    // recorded clips survive a reload while testing.
    const hydratedRef = useRef(false);
    useEffect(() => {
        loadSession().then((s) => {
            if (s && s.length) setClips(s);
            hydratedRef.current = true;
        });
    }, []);
    useEffect(() => {
        if (!hydratedRef.current) return;
        const h = setTimeout(() => void saveSession(clips), 400);
        return () => clearTimeout(h);
    }, [clips]);

    const resetSession = useCallback(() => {
        void clearSession();
        voiceRef.current = null;
        setPhase("idle");
        setClips(DEFAULT_SCRIPT.map((text) => ({ id: newId(), text, pcm: null, durationSec: 0 })));
    }, []);

    const setClipPcm = useCallback((id: string, pcm: Float32Array) => {
        setClips((cs) => cs.map((c) => (c.id === id ? { ...c, pcm, durationSec: pcm.length / SR, peak: peakOf(pcm) } : c)));
    }, []);

    const stopRecording = useCallback(() => {
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        mrRef.current?.state === "recording" && mrRef.current.stop();
        setRecordingId(null);
    }, []);

    const startRecording = useCallback(
        async (id: string) => {
            if (recordingId || phase === "loading" || phase === "encoding") return;
            setErr(null);
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                streamRef.current = stream;
                const mr = new MediaRecorder(stream);
                chunksRef.current = [];
                mr.ondataavailable = (e) => {
                    if (e.data.size) chunksRef.current.push(e.data);
                };
                mr.onstop = async () => {
                    streamRef.current?.getTracks().forEach((t) => t.stop());
                    streamRef.current = null;
                    const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
                    try {
                        const pcm = trimSilence(await decodeToPcm24k(blob));
                        setClipPcm(id, pcm);
                    } catch (e) {
                        setErr(`decode failed: ${e}`);
                    }
                };
                mr.start();
                mrRef.current = mr;
                setRecordingId(id);
                setRecSecs(0);
                timerRef.current = setInterval(() => setRecSecs((s) => s + 0.1), 100);
            } catch (e) {
                setErr(`microphone unavailable: ${e}`);
            }
        },
        [recordingId, phase, setClipPcm],
    );

    const onUpload = useCallback(async (f: File | undefined) => {
        if (!f) return;
        setErr(null);
        try {
            const pcm = trimSilence(await decodeToPcm24k(f));
            setClips((cs) => [...cs, { id: newId(), text: f.name.replace(/\.[^.]+$/, ""), pcm, durationSec: pcm.length / SR }]);
        } catch (e) {
            setErr(`decode failed: ${e}`);
        }
    }, []);

    const recorded = clips.filter((c) => c.pcm);
    const totalSecs = recorded.reduce((n, c) => n + c.durationSec, 0);

    // Live stage progress → progress bar + a small running log (deduped by stage).
    const onProg = useCallback((frac: number, stage: string) => {
        setProcPct(Math.round(frac * 100));
        setProcStage(stage);
        setLogLines((l) => (l.length && l[l.length - 1].endsWith(stage) ? l : [...l, `${(frac * 100) | 0}% · ${stage}`].slice(-8)));
    }, []);

    const createVoice = useCallback(async () => {
        if (recorded.length < MIN_CLIPS || phase === "encoding" || phase === "loading") return;
        setErr(null);
        setPhase("loading");
        setDlPct(0);
        setProcPct(0);
        setProcStage("");
        setLogLines([]);
        try {
            const client = await getSharedClone(styletts2BlobUrl(), STYLETTS2_MODEL.size, (f) => setDlPct(Math.round(f * 100)));
            setPhase("encoding");
            // Encode EACH clip to its own style vector (capped to CLIP_CAP_SEC), then take the
            // robust trimmed mean — dropping noisy/clipped outliers — instead of concatenating
            // every clip into one blob where a single bad take poisons the whole voice.
            const cap = CLIP_CAP_SEC * SR;
            const n = recorded.length;
            const styles: Float32Array[] = [];
            for (let i = 0; i < n; i++) {
                const pcm = recorded[i].pcm!;
                const clip = pcm.length > cap ? pcm.subarray(0, cap) : pcm;
                const v = await client.encodeVoice(clip, (f, s) => onProg((i + f) / n, `clip ${i + 1}/${n} · ${s}`));
                styles.push(v);
            }
            const { vec, dropped } = robustVoice(styles);
            voiceRef.current = vec;
            if (dropped > 0) {
                setLogLines((l) => [...l, `robust average · dropped ${dropped} noisy clip${dropped > 1 ? "s" : ""}`].slice(-8));
            }
            setPhase("done");
        } catch (e) {
            setErr(String(e));
            setPhase("error");
        }
    }, [recorded, phase, onProg]);

    const testVoice = useCallback(async () => {
        if (!voiceRef.current || phase === "synth") return;
        setErr(null);
        setProcPct(0);
        setProcStage("");
        setLogLines([]);
        setPhase("synth");
        try {
            const client = await getSharedClone(styletts2BlobUrl(), STYLETTS2_MODEL.size);
            playPcm(await client.synthesize(testText, voiceRef.current, onProg), client.sampleRate);
        } catch (e) {
            setErr(String(e));
        } finally {
            setPhase("done");
        }
    }, [testText, phase, onProg]);

    const saveVoice = useCallback(() => {
        if (!voiceRef.current) return;
        // copy into a plain ArrayBuffer (cross-origin-isolated lib types Float32Array as
        // ArrayBufferLike-backed, which isn't a valid BlobPart)
        const ab = new ArrayBuffer(voiceRef.current.byteLength);
        new Float32Array(ab).set(voiceRef.current);
        const url = URL.createObjectURL(new Blob([ab], { type: "application/octet-stream" }));
        const a = document.createElement("a");
        a.href = url;
        a.download = `my-voice-${Date.now()}.f32`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, []);

    const busy = phase === "encoding" || phase === "loading";

    return (
        <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
                <AudioLines className="size-4" /> Clone your voice
            </div>
            <p className="text-[11px] text-muted-foreground">
                Read each line aloud into your mic. The clips below are your session — re-record, play,
                delete, add prompts, or upload audio. Record at least {MIN_CLIPS}, then create your voice:
                StyleTTS2 encodes your clips into a voice you can speak any text in. More varied clips ⇒ a
                closer match. Everything runs on your device — desktop only.
            </p>
            {/* Consent / disclosure (required by the StyleTTS2-LibriTTS weight license). */}
            <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-700 dark:text-amber-300">
                Only clone a voice you have permission to use, and disclose that audio is AI-synthesized
                when you share it.
            </p>

            {/* Progress */}
            <div className="flex items-center gap-3 rounded-md border border-border bg-muted/20 px-3 py-2 text-xs">
                <span className="font-medium">
                    {recorded.length} / {clips.length} recorded
                </span>
                <span className="text-muted-foreground">{fmtDur(totalSecs)} total</span>
                <div className="ml-auto h-1.5 w-24 overflow-hidden rounded-full bg-muted">
                    <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${clips.length ? (recorded.length / clips.length) * 100 : 0}%` }}
                    />
                </div>
            </div>

            {/* Editable clip list */}
            <div className="flex flex-col gap-1.5">
                {clips.map((c, i) => {
                    const isRec = recordingId === c.id;
                    return (
                        <div
                            key={c.id}
                            className={cn(
                                "flex items-center gap-2 rounded-md border px-2 py-1.5",
                                isRec ? "border-red-500/60 bg-red-500/10" : c.pcm ? "border-emerald-500/40 bg-emerald-500/5" : "border-border",
                            )}
                        >
                            <span className="w-5 shrink-0 text-center text-[11px] text-muted-foreground">{i + 1}</span>
                            <Input
                                value={c.text}
                                onChange={(e) => setClips((cs) => cs.map((x) => (x.id === c.id ? { ...x, text: e.target.value } : x)))}
                                placeholder="Type a line to read…"
                                disabled={busy}
                                className="h-7 flex-1 text-xs"
                            />
                            <span className="w-14 shrink-0 text-right text-[10px] text-muted-foreground">
                                {isRec ? `● ${mmss(recSecs)}` : c.pcm ? fmtDur(c.durationSec) : "—"}
                            </span>
                            {!isRec && c.pcm && c.peak !== undefined && (c.peak >= 0.99 || c.peak < 0.05) && (
                                <span
                                    className="shrink-0 text-[10px] text-amber-500"
                                    title={c.peak >= 0.99 ? "Clipping — record a bit quieter for a cleaner clone" : "Very quiet — move closer to the mic"}
                                >
                                    {c.peak >= 0.99 ? "⚠ clip" : "⚠ quiet"}
                                </span>
                            )}
                            {isRec ? (
                                <Button size="sm" variant="destructive" className="h-7 px-2" onClick={stopRecording}>
                                    <Square className="size-3" /> Stop
                                </Button>
                            ) : (
                                <Button
                                    size="sm"
                                    variant={c.pcm ? "outline" : "default"}
                                    className="h-7 px-2"
                                    title={c.pcm ? "Re-record" : "Record"}
                                    disabled={!!recordingId || busy || !c.text.trim()}
                                    onClick={() => startRecording(c.id)}
                                >
                                    <Mic className="size-3" /> {c.pcm ? "Redo" : "Rec"}
                                </Button>
                            )}
                            <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 w-7 p-0"
                                title="Play"
                                disabled={!c.pcm || isRec}
                                onClick={() => c.pcm && playPcm(c.pcm, SR)}
                            >
                                <Play className="size-3.5" />
                            </Button>
                            <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                                title="Delete"
                                disabled={isRec || busy}
                                onClick={() => setClips((cs) => cs.filter((x) => x.id !== c.id))}
                            >
                                <Trash2 className="size-3.5" />
                            </Button>
                        </div>
                    );
                })}
            </div>

            {/* Add / upload */}
            <div className="flex flex-wrap items-center gap-2">
                <Button
                    size="sm"
                    variant="outline"
                    className="h-7"
                    disabled={busy}
                    onClick={() => setClips((cs) => [...cs, { id: newId(), text: "", pcm: null, durationSec: 0 }])}
                >
                    <Plus className="size-3.5" /> Add prompt
                </Button>
                <label className={cn("inline-flex h-7 cursor-pointer items-center gap-1 rounded-md border border-border px-2 text-xs hover:bg-muted/50", busy && "pointer-events-none opacity-50")}>
                    <Upload className="size-3.5" /> Upload clip
                    <input type="file" accept="audio/*" className="hidden" disabled={busy} onChange={(e) => onUpload(e.target.files?.[0])} />
                </label>
                <Button size="sm" variant="ghost" className="ml-auto h-7 text-muted-foreground" disabled={busy} onClick={resetSession} title="Clear the saved session and start over">
                    <RotateCcw className="size-3.5" /> Reset
                </Button>
            </div>

            {/* Create voice */}
            <div className="mt-1 flex items-center gap-2 border-t border-border pt-3">
                <Button size="sm" onClick={createVoice} disabled={recorded.length < MIN_CLIPS || busy}>
                    {phase === "done" ? "Re-create voice" : "Create my voice"}
                </Button>
                {recorded.length < MIN_CLIPS && !busy && (
                    <span className="text-[11px] text-muted-foreground">record at least {MIN_CLIPS} clip</span>
                )}
                {busy && (
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Loader2 className="size-3 animate-spin" />
                        {phase === "loading" ? `downloading cloning model… ${dlPct}%` : "encoding your voice…"}
                    </span>
                )}
            </div>
            {phase === "loading" && (
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div className="h-full bg-primary transition-all" style={{ width: `${dlPct}%` }} />
                </div>
            )}

            {/* live stage progress + log for encode / synth (CPU is slow — show everything) */}
            {(phase === "encoding" || phase === "synth" || (logLines.length > 0 && phase === "done")) && (
                <div className="flex flex-col gap-1.5 rounded-md border border-border bg-muted/20 p-2">
                    <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span className="truncate">{procStage || (phase === "synth" ? "synthesizing" : "encoding")}…</span>
                        <span className="tabular-nums">{procPct}%</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                        <div className="h-full bg-primary transition-all" style={{ width: `${procPct}%` }} />
                    </div>
                    {logLines.length > 0 && (
                        <div className="mt-1 max-h-24 overflow-y-auto font-mono text-[10px] leading-tight text-muted-foreground">
                            {logLines.map((l, i) => (
                                <div key={i}>{l}</div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {phase !== "idle" && phase !== "loading" && phase !== "encoding" && voiceRef.current && (
                <div className="flex flex-col gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3">
                    <div className="text-xs font-medium">Your cloned voice is ready — type anything and hear it</div>
                    <Input value={testText} onChange={(e) => setTestText(e.target.value)} className="text-xs" />
                    <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={testVoice} disabled={phase === "synth"}>
                            {phase === "synth" ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                            {phase === "synth" ? "synthesizing…" : "Speak it"}
                        </Button>
                        <Button size="sm" variant="outline" onClick={saveVoice} title="Download a portable .f32 file">
                            Export (.f32)
                        </Button>
                    </div>
                    {/* Save into the in-app library so it appears in the Voice tab. */}
                    <div className="mt-1 flex items-center gap-2 border-t border-emerald-500/30 pt-2">
                        <Input value={voiceName} onChange={(e) => setVoiceName(e.target.value)} placeholder="Name this voice" className="h-7 flex-1 text-xs" />
                        <Button
                            size="sm"
                            onClick={() => {
                                if (!voiceRef.current) return;
                                const v = addVoice(voiceName, voiceRef.current);
                                setSavedMsg(`Saved “${v.name}” — pick it in the Voice tab`);
                            }}
                        >
                            <Save className="size-3.5" /> Save to my voices
                        </Button>
                    </div>
                    {savedMsg && <div className="text-[11px] text-emerald-600 dark:text-emerald-400">{savedMsg}</div>}
                </div>
            )}

            {err && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>}
        </div>
    );
}
