import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ThinkingBlock } from "@/components/ThinkingBlock";
import { type ChatMessage, type ImageAttachment } from "@/lib/types";
import { renderMarkdown } from "@/lib/markdown";
import { parseModelContent } from "@/lib/parseModel";
import { cn } from "@/lib/utils";
import { Mic, Send, Square, Paperclip, X, Volume2 } from "lucide-react";
import { getSharedTts } from "@/lib/tts-client";
import { playPcm } from "@/lib/wav";
import { kokoroBlobUrl } from "@/lib/api";
import { MicButton } from "@/components/MicButton";
import { PipelineProgress, type PipelineProgressState } from "@/components/PipelineProgress";
import type { VoiceOptions } from "@/lib/voice";

// The think-token slips into the chat history when the user enables
// thinking mode. It's a control signal for the model, not user content,
// and must not be rendered.
const THINK_TOKEN = "<|think|>";

interface Props {
    messages:    ChatMessage[];
    canType:     boolean;   // input enabled (model ready, not busy)
    canSend:     boolean;   // Send button enabled (canType AND something to send)
    canStop:     boolean;
    /** Set when the loaded model has the vision tower wired up. Drives
     *  whether the "+" button surfaces an image picker. */
    canAttach:   boolean;
    /** Set when the loaded model has the audio tower wired up.
     *  Drives whether the mic button is shown / enabled. */
    canRecord:   boolean;
    prompt:      string;
    /** Images attached to the next user turn (cleared after send). */
    pendingImages: ImageAttachment[];
    /** Voice clips attached to the next user turn (cleared after send).
     *  Index in this array is the only identity — playback support
     *  comes later. */
    pendingAudio: { durationMs: number }[];
    /** VAD tunables for the mic button (silence cutoff, RMS threshold,
     *  etc.). Surfaced via the Voice section of Settings. */
    voice: VoiceOptions;
    onPromptChange:  (s: string) => void;
    onSend:          () => void;
    onStop:          () => void;
    onAttachFiles:   (files: FileList) => void;
    onRemoveImage:   (idx: number) => void;
    onCaptureAudio:  (pcm: Float32Array) => void | Promise<void>;
    onRemoveAudio:   (idx: number) => void;
    onAudioError?:   (msg: string) => void;
    /** When non-null, a large progress strip mounts above the input row
     *  showing per-layer vision-encode progress for the current image. */
    pipelineProgress?: PipelineProgressState | null;
    statusLine?: string;
    /** Optional drop-in for the empty chat history pane (e.g. a
     *  no-model-loaded card). Falls back to a plain hint when omitted. */
    emptyState?: ReactNode;
    className?:  string;
}

function RoleLabel({ role }: { role: ChatMessage["role"] }) {
    return (
        <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            {role}
        </div>
    );
}

function ImageGrid({ images }: { images: ImageAttachment[] }) {
    return (
        <div className="mb-2 flex flex-wrap gap-1.5">
            {images.map((im, i) => (
                <img
                    key={i}
                    src={im.dataUrl}
                    alt={`attachment ${i + 1}`}
                    className="h-24 w-24 rounded-md border border-border object-cover"
                />
            ))}
        </div>
    );
}

function UserBubble({ content, images }: { content: string; images?: ImageAttachment[] }) {
    const html = useMemo(() => content ? renderMarkdown(content) : "", [content]);
    return (
        <div className="rounded-md border-l-2 border-primary bg-primary/10 p-3 text-sm break-words animate-fade-in">
            <RoleLabel role="user" />
            {images && images.length > 0 && <ImageGrid images={images} />}
            {content ? (
                <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
            ) : !(images && images.length) ? (
                <span className="inline-block animate-pulse text-muted-foreground">▍</span>
            ) : null}
        </div>
    );
}

/** Speak an assistant reply via the shared Kokoro TTS (loads the model on first use). */
function SpeakButton({ text }: { text: string }) {
    const [busy, setBusy] = useState(false);
    const speak = async () => {
        if (busy || !text.trim()) return;
        setBusy(true);
        try {
            const c = await getSharedTts(kokoroBlobUrl());
            const pcm = await c.synthesize(text, "af_heart");
            playPcm(pcm, c.sampleRate);
        } catch {
            /* ignore — TTS load/synth failure shouldn't break chat */
        } finally {
            setBusy(false);
        }
    };
    return (
        <button
            type="button"
            onClick={speak}
            disabled={busy}
            title="Speak this reply"
            className="mt-2 inline-flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
        >
            <Volume2 className={cn("size-3.5", busy && "animate-pulse")} />
            {busy ? "speaking…" : "speak"}
        </button>
    );
}

function ModelBubble({ content }: { content: string }) {
    const parsed = useMemo(() => parseModelContent(content), [content]);
    const html   = useMemo(
        () => parsed.response ? renderMarkdown(parsed.response) : "",
        [parsed.response],
    );
    return (
        <div className="rounded-md border-l-2 border-muted-foreground bg-muted/50 p-3 text-sm break-words animate-fade-in">
            <RoleLabel role="model" />
            {parsed.thinking !== null && (
                <ThinkingBlock
                    text={parsed.thinking}
                    isThinking={parsed.isThinking}
                    isComplete={parsed.isComplete}
                />
            )}
            {parsed.response ? (
                <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
            ) : parsed.thinking === null ? (
                <span className="inline-block animate-pulse text-muted-foreground">▍</span>
            ) : null}
            {parsed.response && parsed.isComplete !== false && <SpeakButton text={parsed.response} />}
        </div>
    );
}

function SystemBubble({ content }: { content: string }) {
    const stripped = content.replaceAll(THINK_TOKEN, "").trim();
    if (!stripped) return null;
    return (
        <div className="rounded-md border-l-2 border-yellow-500 bg-yellow-500/10 p-3 text-sm whitespace-pre-wrap break-words animate-fade-in">
            <RoleLabel role="system" />
            {stripped}
        </div>
    );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
    if (msg.role === "system") return <SystemBubble content={msg.content} />;
    if (msg.role === "model")  return <ModelBubble  content={msg.content} />;
    return <UserBubble content={msg.content} images={msg.images} />;
}

/**
 * Chat panel meant to fill its parent's remaining height. History scrolls
 * internally; input row pins to the bottom. The outer page should NOT scroll.
 */
export function ChatPanel(props: Props) {
    const historyRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    // Collapse the Attach + Mic buttons when the prompt input has focus
    // so the textarea gets the full row width. The buttons animate
    // back in when the input blurs (tap outside, switch tabs, etc.).
    // Standard pattern from Slack / Discord / iMessage compose rows.
    const [inputFocused, setInputFocused] = useState(false);

    // Auto-scroll to the latest message.
    useEffect(() => {
        if (historyRef.current) {
            historyRef.current.scrollTop = historyRef.current.scrollHeight;
        }
    }, [props.messages]);

    // Delegated click handler for `.codeblock-copy` buttons rendered by
    // the markdown pipeline. One listener for the whole history beats a
    // ref per code block.
    useEffect(() => {
        const root = historyRef.current;
        if (!root) return;
        const onClick = async (e: MouseEvent) => {
            const target = e.target as HTMLElement | null;
            const btn = target?.closest<HTMLElement>("[data-bw-copy]");
            if (!btn) return;
            const pre = btn.nextElementSibling as HTMLElement | null;
            const code = pre?.querySelector("code");
            if (!code) return;
            const text = code.textContent ?? "";
            try {
                await navigator.clipboard.writeText(text);
                const prev = btn.textContent;
                btn.textContent = "copied";
                setTimeout(() => { btn.textContent = prev ?? "copy"; }, 1200);
            } catch { /* clipboard denied — silent */ }
        };
        root.addEventListener("click", onClick);
        return () => root.removeEventListener("click", onClick);
    }, []);

    const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey && props.canSend) {
            e.preventDefault();
            props.onSend();
        }
    };

    const onAttachClick = () => {
        fileInputRef.current?.click();
    };

    const onFilesPicked = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) props.onAttachFiles(files);
        // Reset so re-picking the same file re-fires onChange.
        e.target.value = "";
    };

    return (
        <div className={cn("flex h-full min-h-0 flex-col", props.className)}>
            <div
                ref={historyRef}
                className="flex-1 min-h-0 overflow-y-auto px-3 py-2 sm:px-4"
            >
                {props.messages.length === 0 ? (
                    props.emptyState ?? (
                        <p className="mt-8 text-center text-xs text-muted-foreground">
                            Say hi.
                        </p>
                    )
                ) : (
                    <div className="mx-auto flex max-w-3xl flex-col gap-2">
                        {props.messages.map((m, i) => <MessageBubble key={i} msg={m} />)}
                    </div>
                )}
            </div>

            {props.statusLine && (
                <p className="border-t border-border bg-background/60 px-3 py-1 text-[0.65rem] text-muted-foreground sm:px-4">
                    {props.statusLine}
                </p>
            )}

            {props.pipelineProgress && <PipelineProgress state={props.pipelineProgress} />}

            {/* Pending-attachment preview strip, only when there's at
                least one image or voice clip queued. */}
            {(props.pendingImages.length + props.pendingAudio.length) > 0 && (
                <div className="flex flex-wrap gap-1.5 border-t border-border bg-background/60 px-2 py-2 sm:px-3">
                    {props.pendingImages.map((im, i) => (
                        <div
                            key={`img-${i}`}
                            className="relative h-16 w-16 overflow-hidden rounded-md border border-border"
                        >
                            <img src={im.dataUrl} alt={`pending ${i + 1}`} className="h-full w-full object-cover" />
                            <button
                                type="button"
                                onClick={() => props.onRemoveImage(i)}
                                className="absolute right-0.5 top-0.5 rounded-full bg-black/60 p-0.5 text-white hover:bg-black/80"
                                aria-label={`Remove attachment ${i + 1}`}
                                title="Remove"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        </div>
                    ))}
                    {props.pendingAudio.map((a, i) => (
                        <div
                            key={`aud-${i}`}
                            className="relative flex h-16 items-center gap-2 rounded-md border border-border bg-card/60 px-2 text-xs"
                        >
                            <Mic className="h-4 w-4 text-muted-foreground" />
                            <span className="font-mono tabular-nums text-muted-foreground">
                                {(a.durationMs / 1000).toFixed(1)}s
                            </span>
                            <button
                                type="button"
                                onClick={() => props.onRemoveAudio(i)}
                                className="absolute right-0.5 top-0.5 rounded-full bg-black/60 p-0.5 text-white hover:bg-black/80"
                                aria-label={`Remove voice clip ${i + 1}`}
                                title="Remove"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <div className="flex gap-1.5 border-t border-border bg-background/80 px-2 py-2 safe-bottom sm:px-3">
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,audio/*"
                    multiple
                    className="hidden"
                    onChange={onFilesPicked}
                />
                <Input
                    placeholder='Say something…'
                    value={props.prompt}
                    onChange={(e) => props.onPromptChange(e.target.value)}
                    onKeyDown={onKeyDown}
                    onFocus={() => setInputFocused(true)}
                    onBlur={() => setInputFocused(false)}
                    disabled={!props.canType}
                    className="flex-1 min-w-0"
                />
                {props.canStop ? (
                    <Button onClick={props.onStop} variant="destructive" title="Stop"><Square /></Button>
                ) : (
                    <Button onClick={props.onSend} disabled={!props.canSend} title="Send"><Send /></Button>
                )}
                {/* Attach + Mic collapse to width 0 when the input has
                 *  focus, freeing horizontal space for the textarea.
                 *  Send/Stop stay visible because they're needed while
                 *  typing. To attach a file once the buttons are
                 *  hidden, the user just taps outside the input (or
                 *  presses Esc) — the buttons animate back in. */}
                <div
                    className={cn(
                        "flex shrink-0 gap-1.5 overflow-hidden transition-[max-width,opacity] duration-200 ease-out",
                        inputFocused
                            ? "pointer-events-none max-w-0 opacity-0"
                            // Wide enough for Attach + the *recording* mic (Square + a level
                            // number, e.g. "100") — the idle mic is narrower. 7rem clipped the
                            // recording mic's right edge off-screen.
                            : "max-w-[10rem] opacity-100",
                    )}
                >
                    <Button
                        onClick={onAttachClick}
                        disabled={!props.canAttach || !props.canType}
                        variant="outline"
                        title={props.canAttach
                            ? "Attach image or audio file for analysis"
                            : "Multimodal tower unavailable for this model"}
                    >
                        <Paperclip />
                    </Button>
                    {props.canRecord && (
                        <MicButton
                            disabled={!props.canType}
                            voice={props.voice}
                            onCapture={props.onCaptureAudio}
                            onError={props.onAudioError}
                            title="Record voice"
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
