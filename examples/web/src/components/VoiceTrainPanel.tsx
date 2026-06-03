import { useCallback, useRef, useState } from "react";
import { AudioLines, Play, Square, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { kokoroBlobUrl } from "@/lib/api";
import { getSharedTts } from "@/lib/tts-client";
import { decodeToPcm24k, playPcm } from "@/lib/wav";

const DEFAULT_REF = "Hello, how are you today?";
const MAX_STEPS = 60;

/** Gradient-free voice training (clone a target speaker's timbre into a voicepack). */
export function VoiceTrainPanel() {
    const [targetPcm, setTargetPcm] = useState<Float32Array | null>(null);
    const [targetName, setTargetName] = useState<string>("");
    const [refText, setRefText] = useState(DEFAULT_REF);
    const [phase, setPhase] = useState<"idle" | "loading" | "training" | "done" | "error">("idle");
    const [step, setStep] = useState(0);
    const [loss, setLoss] = useState<number[]>([]);
    const [err, setErr] = useState<string | null>(null);
    const [testText, setTestText] = useState("This is my trained voice.");
    const stopRef = useRef(false);
    const voiceRef = useRef<Float32Array | null>(null);

    const onFile = useCallback(async (f: File | undefined) => {
        if (!f) return;
        setErr(null);
        try {
            const pcm = await decodeToPcm24k(f);
            setTargetPcm(pcm);
            setTargetName(`${f.name} (${(pcm.length / 24000).toFixed(1)}s)`);
        } catch (e) {
            setErr(`decode failed: ${e}`);
        }
    }, []);

    const train = useCallback(async () => {
        if (!targetPcm || phase === "training") return;
        setErr(null);
        setPhase("loading");
        setLoss([]);
        setStep(0);
        stopRef.current = false;
        try {
            const c = await getSharedTts(kokoroBlobUrl());
            await c.trainBegin(targetPcm, refText.trim() || DEFAULT_REF);
            setPhase("training");
            for (let i = 0; i < MAX_STEPS; i++) {
                if (stopRef.current) break;
                const l = await c.trainStep();
                setStep(i + 1);
                setLoss((cur) => [...cur, l]);
            }
            voiceRef.current = await c.trainedVoice();
            setPhase("done");
        } catch (e) {
            setErr(String(e));
            setPhase("error");
        }
    }, [targetPcm, refText, phase]);

    const testVoice = useCallback(async () => {
        if (!voiceRef.current) return;
        try {
            const c = await getSharedTts(kokoroBlobUrl());
            const pcm = await c.synthesizeWithVoice(testText, voiceRef.current);
            playPcm(pcm, c.sampleRate);
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
        const blob = new Blob([ab], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `kokoro-voice-${Date.now()}.f32`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, []);

    const first = loss[0];
    const last = loss[loss.length - 1];
    const reduction = first && last ? Math.round((1 - last / first) * 100) : 0;

    return (
        <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
                <AudioLines className="size-4" /> Voice training — clone a speaker into a voicepack
            </div>
            <p className="text-[11px] text-muted-foreground">
                Upload a short clip of a target voice. Gradient-free optimization shifts a Kokoro voicepack
                toward that speaker's timbre. (v1: timbre match; synthesis is slow on weak GPUs.)
            </p>

            <div className="flex flex-wrap items-center gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-border px-2 py-1 text-xs hover:bg-muted/50">
                    <Upload className="size-3.5" /> Target clip
                    <input type="file" accept="audio/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
                </label>
                {targetName && <span className="text-xs text-muted-foreground">{targetName}</span>}
            </div>

            <label className="text-xs text-muted-foreground">
                Reference text (re-synthesized each step)
                <Input value={refText} onChange={(e) => setRefText(e.target.value)} className="mt-1" />
            </label>

            <div className="flex items-center gap-2">
                {phase === "training" || phase === "loading" ? (
                    <Button size="sm" variant="destructive" onClick={() => (stopRef.current = true)}>
                        <Square className="size-3.5" /> Stop
                    </Button>
                ) : (
                    <Button size="sm" onClick={train} disabled={!targetPcm}>
                        Start training
                    </Button>
                )}
                {(phase === "training" || phase === "loading") && (
                    <span className="text-xs text-muted-foreground">
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
                    <div className="text-xs font-medium">Trained voice ready</div>
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
