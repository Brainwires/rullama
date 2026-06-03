import { useCallback, useRef, useState } from "react";
import { AudioLines, Download, Loader2, Play, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { KOKORO_MODEL, kokoroBlobUrl } from "@/lib/api";
import { getSharedTts, type TtsClient, type TtsClip } from "@/lib/tts-client";
import { cn } from "@/lib/utils";
import { downloadWav, playPcm } from "@/lib/wav";

// Only af_heart is bundled in the current Kokoro GGUF; add more voicepacks to grow this.
const VOICES = ["af_heart"];

type LoadState = "idle" | "loading" | "ready" | "error";

export function VoicePanel() {
    const client = useRef<TtsClient | null>(null);
    const [load, setLoad] = useState<LoadState>("idle");
    const [loadPct, setLoadPct] = useState(0);
    const [err, setErr] = useState<string | null>(null);
    const [text, setText] = useState("Hello, this is text to speech running entirely in your browser.");
    const [voice, setVoice] = useState(VOICES[0]);
    const [busy, setBusy] = useState(false);
    const [clips, setClips] = useState<TtsClip[]>([]);
    const [active, setActive] = useState<number>(-1);

    const ensureLoaded = useCallback(async () => {
        if (client.current) return;
        setLoad("loading");
        setErr(null);
        try {
            const c = await getSharedTts(kokoroBlobUrl(), (f) => setLoadPct(Math.round(f * 100)));
            client.current = c;
            setLoad("ready");
        } catch (e) {
            setErr(String(e));
            setLoad("error");
        }
    }, []);

    const generate = useCallback(async () => {
        if (!client.current || busy || !text.trim()) return;
        setBusy(true);
        setErr(null);
        try {
            const pcm = await client.current.synthesize(text.trim(), voice);
            const clip: TtsClip = { pcm, sampleRate: client.current.sampleRate, text: text.trim(), voice, ts: Date.now() };
            setClips((cs) => [clip, ...cs]);
            setActive(0);
            playPcm(pcm, clip.sampleRate);
        } catch (e) {
            setErr(String(e));
        } finally {
            setBusy(false);
        }
    }, [busy, text, voice]);

    return (
        <div className="flex h-full min-h-0">
            {/* artifact sidebar */}
            <aside className="hidden w-64 shrink-0 flex-col border-r border-border bg-muted/20 sm:flex">
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
            </aside>

            {/* main */}
            <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                    <AudioLines className="size-4" /> Voice — Kokoro TTS
                </div>

                <Textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type text to speak…"
                    className="min-h-32 resize-none"
                />

                <div className="flex flex-wrap items-center gap-2">
                    <select
                        value={voice}
                        onChange={(e) => setVoice(e.target.value)}
                        className="h-8 rounded-md border border-border bg-background px-2 text-xs"
                    >
                        {VOICES.map((v) => (
                            <option key={v} value={v}>{v}</option>
                        ))}
                    </select>

                    {load !== "ready" ? (
                        <Button size="sm" onClick={ensureLoaded} disabled={load === "loading"}>
                            {load === "loading" ? <><Loader2 className="size-3.5 animate-spin" /> Loading… {loadPct}%</> : "Load voice model"}
                        </Button>
                    ) : (
                        <Button size="sm" onClick={generate} disabled={busy || !text.trim()}>
                            {busy ? <><Loader2 className="size-3.5 animate-spin" /> Generating…</> : "Generate speech"}
                        </Button>
                    )}

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

                {load === "loading" && <Progress value={loadPct} className="h-1" />}
                {err && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>}
                <p className="text-[11px] text-muted-foreground">
                    First load downloads the {Math.round(KOKORO_MODEL.size / 1e6)} MB voice model. Synthesis runs on your GPU via WebGPU.
                </p>
            </div>
        </div>
    );
}
