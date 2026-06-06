import { Loader2, Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMicCapture } from "@/lib/useMicCapture";
import type { VoiceOptions } from "@/lib/voice";

interface Props {
    disabled?: boolean;
    /** VAD tunables (silence cutoff, RMS threshold, etc.) — driven by
     *  the Voice section of Settings. */
    voice: VoiceOptions;
    /** Called with 16 kHz mono f32 PCM once VAD auto-stops. The caller
     *  decides what to do with it (attach to next message, transcribe,
     *  etc.). */
    onCapture: (pcm: Float32Array) => void | Promise<void>;
    onError?: (msg: string) => void;
    title?: string;
}

/**
 * Microphone button with three visible states:
 *
 *   idle:      mic icon, click to start
 *   recording: stop icon + level chip, click to abort
 *   encoding:  spinner, disabled (audio tower is running)
 *
 * VAD-driven auto-stop on silence is handled by `useMicCapture`.
 */
export function MicButton({ disabled, voice, onCapture, onError, title }: Props) {
    const { state, rmsDb, start, cancel } = useMicCapture({
        voice,
        onComplete: onCapture,
        onError: (e) => onError?.(e.message),
    });

    if (state === "recording") {
        // Map dBFS roughly to a 0–100 % level for the badge. Speech
        // peaks around -25 to -10 dB, so the visible range is [-50,0].
        const level = Math.max(0, Math.min(100, ((rmsDb + 50) / 50) * 100));
        return (
            <Button
                type="button"
                onClick={cancel}
                variant="destructive"
                title="Stop recording (cancels VAD auto-stop)"
                aria-label="Stop recording"
                className="font-mono tabular-nums"
            >
                <Square />
                <span className="ml-1 text-[10px]">{level.toFixed(0)}</span>
            </Button>
        );
    }

    if (state === "encoding") {
        return (
            <Button
                type="button"
                disabled
                variant="outline"
                title="Encoding audio…"
                aria-label="Encoding audio"
            >
                <Loader2 className="animate-spin" />
            </Button>
        );
    }

    return (
        <Button
            type="button"
            onClick={start}
            disabled={disabled}
            variant="outline"
            title={title ?? "Record voice"}
            aria-label="Record voice"
        >
            <Mic />
        </Button>
    );
}
