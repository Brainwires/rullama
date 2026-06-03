import { useCallback, useEffect, useRef, useState } from "react";
import { AudioLines, Loader2, Mic, Play, Plus, Square, Trash2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { kokoroBlobUrl } from "@/lib/api";
import { getSharedTts } from "@/lib/tts-client";
import { cn } from "@/lib/utils";
import { decodeToPcm24k, playPcm } from "@/lib/wav";

const MAX_STEPS = 60;
const MIN_CLIPS = 2;
const SR = 24000;

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
}

let _cid = 0;
const newId = () => `c${++_cid}`;

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

const fmtDur = (s: number) => `${s.toFixed(1)}s`;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/**
 * Guided voice training: record your own voice over a short script, edit the clip
 * list (re-record / play / delete / add / upload), then clone your timbre into a
 * Kokoro voicepack. Clips concatenate into the gradient-free optimizer — the timbre
 * signature is timing-invariant, so more clips ⇒ a better target.
 */
export function VoiceTrainPanel() {
    const [clips, setClips] = useState<SessionClip[]>(() =>
        DEFAULT_SCRIPT.map((text) => ({ id: newId(), text, pcm: null, durationSec: 0 })),
    );
    const [recordingId, setRecordingId] = useState<string | null>(null);
    const [recSecs, setRecSecs] = useState(0);
    const [phase, setPhase] = useState<"idle" | "loading" | "training" | "done" | "error">("idle");
    const [step, setStep] = useState(0);
    const [loss, setLoss] = useState<number[]>([]);
    const [err, setErr] = useState<string | null>(null);
    const [testText, setTestText] = useState("This is my trained voice.");

    const mrRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const stopRef = useRef(false);
    const voiceRef = useRef<Float32Array | null>(null);

    // Cleanup any live mic stream / timer on unmount.
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            streamRef.current?.getTracks().forEach((t) => t.stop());
        };
    }, []);

    const setClipPcm = useCallback((id: string, pcm: Float32Array) => {
        setClips((cs) => cs.map((c) => (c.id === id ? { ...c, pcm, durationSec: pcm.length / SR } : c)));
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
            if (recordingId || phase === "training") return;
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

    const train = useCallback(async () => {
        if (recorded.length < MIN_CLIPS || phase === "training") return;
        setErr(null);
        setPhase("loading");
        setLoss([]);
        setStep(0);
        stopRef.current = false;
        try {
            // Concatenate every recorded clip — voice_signature() is mean-log-mel over all
            // frames, so this is the aggregate timbre across the whole session.
            const total = recorded.reduce((n, c) => n + c.pcm!.length, 0);
            const concat = new Float32Array(total);
            let o = 0;
            for (const c of recorded) {
                concat.set(c.pcm!, o);
                o += c.pcm!.length;
            }
            const refText = recorded[0].text.trim() || DEFAULT_SCRIPT[0];
            const client = await getSharedTts(kokoroBlobUrl());
            await client.trainBegin(concat, refText);
            setPhase("training");
            for (let i = 0; i < MAX_STEPS; i++) {
                if (stopRef.current) break;
                const l = await client.trainStep();
                setStep(i + 1);
                setLoss((cur) => [...cur, l]);
            }
            voiceRef.current = await client.trainedVoice();
            setPhase("done");
        } catch (e) {
            setErr(String(e));
            setPhase("error");
        }
    }, [recorded, phase]);

    const testVoice = useCallback(async () => {
        if (!voiceRef.current) return;
        try {
            const client = await getSharedTts(kokoroBlobUrl());
            playPcm(await client.synthesizeWithVoice(testText, voiceRef.current), client.sampleRate);
        } catch (e) {
            setErr(String(e));
        }
    }, [testText]);

    const saveVoice = useCallback(() => {
        if (!voiceRef.current) return;
        // copy into a plain ArrayBuffer (cross-origin-isolated lib types Float32Array as
        // ArrayBufferLike-backed, which isn't a valid BlobPart)
        const ab = new ArrayBuffer(voiceRef.current.byteLength);
        new Float32Array(ab).set(voiceRef.current);
        const url = URL.createObjectURL(new Blob([ab], { type: "application/octet-stream" }));
        const a = document.createElement("a");
        a.href = url;
        a.download = `kokoro-voice-${Date.now()}.f32`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, []);

    const busy = phase === "training" || phase === "loading";
    const first = loss[0];
    const last = loss[loss.length - 1];
    const reduction = first && last ? Math.round((1 - last / first) * 100) : 0;

    return (
        <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
                <AudioLines className="size-4" /> Train your voice
            </div>
            <p className="text-[11px] text-muted-foreground">
                Read each line aloud into your mic. The clips below are your session — re-record, play, or
                delete any of them, add your own prompts, or upload existing audio. Record at least{" "}
                {MIN_CLIPS} clips, then train to clone your timbre into a Kokoro voicepack. More varied
                clips ⇒ a closer match. (Timbre clone, not studio quality.)
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
            </div>

            {/* Train */}
            <div className="mt-1 flex items-center gap-2 border-t border-border pt-3">
                {busy ? (
                    <Button size="sm" variant="destructive" onClick={() => (stopRef.current = true)}>
                        <Square className="size-3.5" /> Stop training
                    </Button>
                ) : (
                    <Button size="sm" onClick={train} disabled={recorded.length < MIN_CLIPS}>
                        {phase === "done" ? "Re-train voice" : "Train voice"}
                    </Button>
                )}
                {recorded.length < MIN_CLIPS && !busy && (
                    <span className="text-[11px] text-muted-foreground">record at least {MIN_CLIPS} clips</span>
                )}
                {busy && (
                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Loader2 className="size-3 animate-spin" />
                        {phase === "loading" ? "loading model…" : `step ${step}/${MAX_STEPS}`}
                        {last != null && ` · loss ${last.toExponential(2)}`}
                    </span>
                )}
            </div>

            {loss.length > 0 && (
                <div className="rounded-md border border-border bg-muted/20 p-2 text-xs">
                    loss {first?.toExponential(2)} → {last?.toExponential(2)}{" "}
                    <span className={reduction > 0 ? "text-emerald-500" : "text-muted-foreground"}>({reduction}% reduction)</span>
                    <div className="mt-1 flex h-8 items-end gap-px">
                        {loss.map((l, i) => (
                            <div key={i} className="flex-1 bg-primary/60" style={{ height: `${first ? Math.max(2, (l / first) * 100) : 2}%` }} />
                        ))}
                    </div>
                </div>
            )}

            {phase === "done" && voiceRef.current && (
                <div className="flex flex-col gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3">
                    <div className="text-xs font-medium">Your trained voice is ready</div>
                    <Input value={testText} onChange={(e) => setTestText(e.target.value)} className="text-xs" />
                    <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={testVoice}>
                            <Play className="size-3.5" /> Test
                        </Button>
                        <Button size="sm" variant="outline" onClick={saveVoice}>
                            Save voicepack (.f32)
                        </Button>
                    </div>
                </div>
            )}

            {err && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>}
        </div>
    );
}
