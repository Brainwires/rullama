import { useState } from "react";
import { EnvironmentStatus } from "@/components/EnvironmentStatus";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { type ChatMessage, type SamplingOptions, DEFAULT_SAMPLING } from "@/lib/types";

export function App() {
    // Model load state
    const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
    const [loadingPercent, setLoadingPercent] = useState(0);
    const [loadingLabel, setLoadingLabel]     = useState("");
    const [statusText, setStatusText]         = useState("no model");

    // Chat state
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [prompt, setPrompt]     = useState("");
    const [busy, setBusy]         = useState(false);

    // Settings
    const [systemPrompt, setSystemPrompt] = useState("");
    const [sampling, setSampling]         = useState<SamplingOptions>(DEFAULT_SAMPLING);
    const [maxTokens, setMaxTokens]       = useState(256);

    const onLoad = (_m: import("@/lib/api").ModelEntry) => {
        // Phase 4 wires this to the inference worker. For now, simulate progress.
        setModelStatus("loading");
        setLoadingPercent(0);
        setLoadingLabel("(phase 3 stub — inference wiring lands next commit)");
        setStatusText("loading…");
        // No-op
    };

    const onSend = () => {
        const text = prompt.trim();
        if (!text) return;
        setMessages([...messages, { role: "user", content: text }]);
        setPrompt("");
    };

    return (
        <div className="min-h-screen bg-background safe-top">
            <div className="mx-auto max-w-3xl px-4 py-6 sm:py-10">
                <header className="mb-6">
                    <h1 className="text-2xl font-semibold tracking-tight">rullama</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Gemma 4 in your browser. Pure Rust → WebAssembly + WebGPU. No server.
                    </p>
                </header>

                <div className="space-y-6">
                    <EnvironmentStatus />
                    <ModelLoader
                        status={modelStatus}
                        loadingPercent={loadingPercent}
                        loadingLabel={loadingLabel}
                        statusText={statusText}
                        onLoad={onLoad}
                    />
                    <SettingsDialog
                        systemPrompt={systemPrompt}
                        onSystemPromptChange={setSystemPrompt}
                        sampling={sampling}
                        onSamplingChange={setSampling}
                        maxTokens={maxTokens}
                        onMaxTokensChange={setMaxTokens}
                    />
                    <ChatPanel
                        messages={messages}
                        canSend={modelStatus === "ready" && !busy && prompt.trim().length > 0}
                        canStop={busy}
                        canReset={modelStatus === "ready" && messages.length > 0 && !busy}
                        prompt={prompt}
                        onPromptChange={setPrompt}
                        onSend={() => { setBusy(true); onSend(); setBusy(false); }}
                        onStop={() => setBusy(false)}
                        onReset={() => setMessages([])}
                    />
                </div>

                <footer className="mt-10 text-center text-xs text-muted-foreground safe-bottom">
                    <a
                        href="https://github.com/Brainwires/rullama"
                        target="_blank"
                        rel="noreferrer"
                        className="hover:text-foreground"
                    >
                        Brainwires / rullama
                    </a>
                </footer>
            </div>
        </div>
    );
}
