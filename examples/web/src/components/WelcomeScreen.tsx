import { cn } from "@/lib/utils";
import iconUrl from "../assets/icon-192.png";

interface Props {
    modelName: string;
    onSuggest: (prompt: string) => void;
    className?: string;
}

const SUGGESTIONS = [
    {
        title: "Tell me a story",
        body:  "about a curious little star and the wind",
        prompt: "Tell me a short story about a curious little star and the wind.",
    },
    {
        title: "Write a haiku",
        body:  "about WebGPU",
        prompt: "Write a haiku about WebGPU.",
    },
    {
        title: "Explain quantization",
        body:  "like I'm new to ML",
        prompt: "Explain Q4_K_M quantization in simple terms, like I'm new to ML.",
    },
];

/** Centered greeting + suggestion chips. Shown when the model is loaded
 *  but no chat is in progress. Adapted (loosely) from the studio's
 *  WelcomeMessage pattern — no logo / typewriter dependency. */
export function WelcomeScreen({ modelName, onSuggest, className }: Props) {
    return (
        <div className={cn(
            "flex h-full min-h-0 flex-col items-center justify-center px-4 py-6 text-center",
            "animate-fade-in",
            className,
        )}>
            <img
                src={iconUrl}
                alt="rullama"
                className="size-20 sm:size-24"
                draggable={false}
            />

            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                rullama
            </h1>
            <p className="mt-1 max-w-sm text-xs text-muted-foreground sm:text-sm">
                Gemma 4 in your browser. Pure Rust → WebAssembly → WebGPU. No server.
            </p>
            <p className="mt-3 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                ready · <span className="text-foreground/80">{modelName}</span>
            </p>

            <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-3">
                {SUGGESTIONS.map((s) => (
                    <button
                        key={s.title}
                        type="button"
                        onClick={() => onSuggest(s.prompt)}
                        className="group flex flex-col gap-0.5 rounded-md border border-border bg-card/50 p-3 text-left text-xs transition-colors hover:bg-card hover:border-primary/40 animate-slide-up"
                    >
                        <span className="font-medium text-foreground">{s.title}</span>
                        <span className="text-[0.7rem] text-muted-foreground group-hover:text-foreground/70">
                            {s.body}
                        </span>
                    </button>
                ))}
            </div>
        </div>
    );
}
