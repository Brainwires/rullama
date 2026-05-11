import { useCallback, useRef, useState } from "react";
import { EnvironmentStatus } from "@/components/EnvironmentStatus";
import { ModelLoader, ModelLoadProgress, type ModelStatus } from "@/components/ModelLoader";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Button } from "@/components/ui/button";
import { type ChatMessage, type SamplingOptions, DEFAULT_SAMPLING } from "@/lib/types";
import { type ModelEntry, blobUrl, beacon } from "@/lib/api";
import { ensureModel, opfsSupported, requestPersistent } from "@/lib/opfs";
import { getClient } from "@/lib/inference";
import { fmtBytes } from "@/lib/utils";
import { Settings2 } from "lucide-react";

const isMobileUA = () => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const THINK_TOKEN = "<|think|>";

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
    const [statusLine, setStatusLine] = useState<string | undefined>();

    // Settings
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [systemPrompt, setSystemPrompt] = useState("");
    const [sampling, setSampling]         = useState<SamplingOptions>(DEFAULT_SAMPLING);
    const [maxTokens, setMaxTokens]       = useState(1024);
    const [thinking, setThinking]         = useState(false);

    const cancelRef = useRef(false);

    const onLoad = useCallback(async (m: ModelEntry) => {
        const client = getClient();
        setModelStatus("loading");
        setLoadingPercent(0);
        setLoadingLabel("checking OPFS…");
        setStatusText("loading…");

        try {
            if (!(await opfsSupported())) throw new Error("OPFS not supported in this browser");
            await requestPersistent();

            const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
            const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";
            const url = blobUrl(m.name);

            const t0 = performance.now();
            const { totalBytes, fromCache } = await ensureModel(url, modelKey, filename, ({ bytesWritten, totalBytes }) => {
                if (totalBytes > 0) {
                    setLoadingPercent((bytesWritten / totalBytes) * 100);
                    const elapsed = (performance.now() - t0) / 1000;
                    const rate = bytesWritten / Math.max(elapsed, 0.001);
                    setLoadingLabel(`${fmtBytes(bytesWritten)} / ${fmtBytes(totalBytes)} — ${fmtBytes(rate)}/s`);
                } else {
                    setLoadingLabel(fmtBytes(bytesWritten));
                }
            });

            if (fromCache) {
                beacon("chat", `OPFS cache hit (${fmtBytes(totalBytes)})`);
            } else {
                beacon("chat", `downloaded ${fmtBytes(totalBytes)} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
            }

            setLoadingLabel("loading into wasm…");
            const mobile = isMobileUA();
            const mobileMaxCtx = 512;
            await client.load(modelKey, filename, {
                maxContext: mobile ? mobileMaxCtx : 0,
                textOnly:   mobile,
            });

            setModelStatus("ready");
            setStatusText(`${m.name}${fromCache ? " ⚡" : ""}`);
            setLoadingLabel("");
            setMessages([]);
        } catch (e) {
            const err = (e as Error).message ?? String(e);
            setModelStatus("error");
            setStatusText(`load failed: ${err}`);
            setLoadingLabel("");
        }
    }, []);

    const onSend = useCallback(async () => {
        if (modelStatus !== "ready" || busy) return;
        const text = prompt.trim();
        if (!text) return;
        const client = getClient();
        cancelRef.current = false;
        setBusy(true);
        setPrompt("");
        setStatusLine(undefined);

        // Compose the system message. When thinking mode is on, prepend
        // <|think|> to whatever the user wrote (or stand alone if empty).
        // This is silent — the displayed history below uses `messages`,
        // which has no system entry to begin with.
        const sysContent = thinking
            ? (systemPrompt.trim() ? `${THINK_TOKEN}${systemPrompt.trim()}` : THINK_TOKEN)
            : systemPrompt.trim();

        // Build the new history list with the user turn appended.
        const history: ChatMessage[] = [
            ...(sysContent && messages.length === 0
                ? [{ role: "system" as const, content: sysContent }]
                : []),
            ...messages,
            { role: "user", content: text },
            { role: "model", content: "" },
        ];
        setMessages(history);

        try {
            await client.setSampling(sampling);
            await client.reset();   // simple-correct: re-feed full history each Send
            const rendered = await client.renderChat(history.slice(0, -1), false);
            const ids = await client.encode(rendered);

            // Prompt-eval
            const t0 = performance.now();
            let next = 0;
            for (let i = 0; i < ids.length; i++) {
                if (cancelRef.current) throw new Error("cancelled");
                next = await client.step(ids[i]);
            }
            const peMs = performance.now() - t0;

            // Generation
            const tg0 = performance.now();
            let emitted = 0;
            let curStr   = (await client.tokenStr(next)) ?? "";
            let curIsEos = await client.isEos(next);
            for (let i = 0; i < maxTokens; i++) {
                if (cancelRef.current) break;
                if (curIsEos) break;
                const piece = curStr.replaceAll("▁", " ");
                history[history.length - 1].content += piece;
                setMessages([...history]);
                emitted++;
                const r = await client.stepAndDecode(next);
                next     = r.next;
                curStr   = r.str ?? "";
                curIsEos = r.isEos;
            }
            const dt = performance.now() - tg0;
            const tps = emitted > 0 ? (emitted * 1000 / dt) : 0;
            setStatusLine(`pe ${peMs.toFixed(0)} ms · gen ${emitted} tok in ${dt.toFixed(0)} ms · ${tps.toFixed(2)} tok/s`);
            beacon("chat", `gen ${emitted} tok in ${dt.toFixed(0)} ms (${tps.toFixed(2)} tok/s)`);
        } catch (e) {
            setStatusLine(`error: ${(e as Error).message}`);
        } finally {
            setBusy(false);
        }
    }, [busy, maxTokens, messages, modelStatus, prompt, sampling, systemPrompt, thinking]);

    const onReset = useCallback(() => {
        if (busy) return;
        setMessages([]);
        setStatusLine(undefined);
        void getClient().reset();
    }, [busy]);

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* Compact top toolbar — title, env, model picker, settings. */}
            <header className="flex flex-wrap items-center gap-2 border-b border-border bg-card/40 px-3 py-1.5 text-xs safe-top">
                <span className="font-semibold tracking-tight">rullama</span>
                <EnvironmentStatus />
                <div className="ml-auto flex flex-wrap items-center gap-1">
                    <ModelLoader
                        status={modelStatus}
                        loadingPercent={loadingPercent}
                        loadingLabel={loadingLabel}
                        statusText={statusText}
                        onLoad={onLoad}
                    />
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => setSettingsOpen(!settingsOpen)}
                        title="Toggle settings"
                        aria-pressed={settingsOpen}
                    >
                        <Settings2 />
                    </Button>
                </div>
            </header>

            {modelStatus === "loading" && (
                <ModelLoadProgress percent={loadingPercent} label={loadingLabel} />
            )}

            {settingsOpen && (
                <SettingsDialog
                    systemPrompt={systemPrompt}
                    onSystemPromptChange={setSystemPrompt}
                    sampling={sampling}
                    onSamplingChange={setSampling}
                    maxTokens={maxTokens}
                    onMaxTokensChange={setMaxTokens}
                    thinking={thinking}
                    onThinkingChange={setThinking}
                />
            )}

            <ChatPanel
                messages={messages}
                // Typing stays enabled during generation so the user can
                // queue up the next message. Send is still gated by `!busy`.
                canType={modelStatus === "ready"}
                canSend={modelStatus === "ready" && !busy && prompt.trim().length > 0}
                canStop={busy}
                canReset={modelStatus === "ready" && messages.length > 0 && !busy}
                prompt={prompt}
                onPromptChange={setPrompt}
                onSend={onSend}
                onStop={() => { cancelRef.current = true; }}
                onReset={onReset}
                statusLine={statusLine}
            />
        </div>
    );
}
