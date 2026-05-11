import { useCallback, useEffect, useRef, useState } from "react";
import { ModelLoadProgress, type ModelStatus } from "@/components/ModelLoader";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsDialog, SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { ConversationList } from "@/components/ConversationList";
import { Button } from "@/components/ui/button";
import { type ChatMessage, type SamplingOptions, DEFAULT_SAMPLING } from "@/lib/types";
import { type ModelEntry, blobUrl, beacon } from "@/lib/api";
import { ensureModel, opfsSupported, requestPersistent, wipeModel } from "@/lib/opfs";
import { getClient, type ConversationRow } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { usePersistedState } from "@/lib/persisted";
import { fmtBytes, clampInt, clampNum } from "@/lib/utils";
import { Settings, History } from "lucide-react";

const isMobileUA = () => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const THINK_TOKEN = "<|think|>";
const TITLE_MAX_LEN = 40;

function suggestTitle(text: string): string {
    const t = text.trim().replace(/\s+/g, " ");
    return t.length <= TITLE_MAX_LEN ? t : t.slice(0, TITLE_MAX_LEN - 1) + "…";
}

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

    // Conversation persistence (rsqlite-wasm)
    const [conversations, setConversations]     = useState<ConversationRow[]>([]);
    const [activeConvId, setActiveConvId]       = useState<string | null>(null);
    const [historyOpen, setHistoryOpen]         = useState(false);

    // Settings — opens by default since first mount is always "no model".
    // We don't auto-close on load; user dismisses with ⚙. Open/closed
    // state is transient (per page load).
    const [settingsOpen, setSettingsOpen] = useState(true);

    // The four user-tunable settings persist across reloads via localStorage.
    const [systemPrompt, setSystemPrompt] = usePersistedState<string>("systemPrompt", "");
    const [sampling,     setSampling]     = usePersistedState<SamplingOptions>("sampling", DEFAULT_SAMPLING);
    const [maxTokens,    setMaxTokens]    = usePersistedState<number>("maxTokens", 1024);
    const [thinking,     setThinking]     = usePersistedState<boolean>("thinking", false);

    const cancelRef = useRef(false);
    const { showToast, dismissToast } = useToast();

    // One-time sanitization of persisted values. Catches localStorage
    // entries from older versions (or hand-edited values) that fall
    // outside current bounds — the slider clamps cover fresh edits.
    useEffect(() => {
        const B = SETTINGS_BOUNDS;
        const next: SamplingOptions = {
            temperature:        clampNum(sampling.temperature, B.temperature.min, B.temperature.max, B.temperature.fallback),
            top_k:              clampInt(sampling.top_k,       B.top_k.min,       B.top_k.max,       B.top_k.fallback),
            top_p:              clampNum(sampling.top_p,       B.top_p.min,       B.top_p.max,       B.top_p.fallback),
            repetition_penalty: clampNum(sampling.repetition_penalty, B.repetition_penalty.min, B.repetition_penalty.max, B.repetition_penalty.fallback),
            seed:               Number.isFinite(sampling.seed) ? sampling.seed : 0,
        };
        if (next.temperature !== sampling.temperature
            || next.top_k !== sampling.top_k
            || next.top_p !== sampling.top_p
            || next.repetition_penalty !== sampling.repetition_penalty
            || next.seed !== sampling.seed) {
            setSampling(next);
        }
        const mt = clampInt(maxTokens, B.maxTokens.min, B.maxTokens.max, B.maxTokens.fallback);
        if (mt !== maxTokens) setMaxTokens(mt);
        // Run once on mount with the hydrated values.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Probe environment once on mount. Failures become sticky toasts the
    // user must dismiss (vs the prior compact pill row that was always
    // visible). Successful capabilities are silent.
    useEffect(() => {
        const has = (k: () => boolean) => { try { return k(); } catch { return false; } };
        if (!has(() => typeof navigator !== "undefined" && "gpu" in navigator)) {
            showToast({
                id: "env-webgpu",
                level: "error",
                title: "WebGPU not available",
                message: "rullama cannot run inference without WebGPU. On iOS, update to iOS 18+. On desktop, use a recent Chrome / Edge / Safari.",
                persist: true,
            });
        }
        if (!has(() => typeof navigator !== "undefined"
                       && !!navigator.storage
                       && typeof navigator.storage.getDirectory === "function")) {
            showToast({
                id: "env-opfs",
                level: "error",
                title: "OPFS not available",
                message: "Models larger than ~3 GB need the Origin Private File System for streaming. Without it, large GGUFs will OOM the page.",
                persist: true,
            });
        }
        if (!has(() => typeof window !== "undefined" && window.crossOriginIsolated)) {
            showToast({
                id: "env-coi",
                level: "warn",
                title: "Cross-origin isolation off",
                message: "Required for SharedArrayBuffer / high-res timers. Make sure the page is served with COOP+COEP headers.",
                persist: true,
            });
        }
    }, [showToast]);

    // Bootstrap DB + conversation list on mount.
    useEffect(() => {
        const client = getClient();
        (async () => {
            try {
                await client.dbInit();
                const rows = await client.convList();
                setConversations(rows);
            } catch (e) {
                showToast({
                    level: "error",
                    title: "Database init failed",
                    message: (e as Error).message,
                });
            }
        })();
    }, [showToast]);

    const refreshConversations = useCallback(async () => {
        try {
            const rows = await getClient().convList();
            setConversations(rows);
        } catch (e) {
            showToast({ level: "warn", title: "Could not refresh conversations", message: (e as Error).message });
        }
    }, [showToast]);

    const onSelectConversation = useCallback(async (id: string) => {
        if (busy) return;
        try {
            const rows = await getClient().msgList(id);
            // Ignore any legacy system rows on load — system content is
            // derived per-send from current settings, not from history.
            const ms: ChatMessage[] = rows
                .filter((r) => r.role === "user" || r.role === "model")
                .map((r) => ({ role: r.role as ChatMessage["role"], content: r.content }));
            setMessages(ms);
            setActiveConvId(id);
            setStatusLine(undefined);
        } catch (e) {
            showToast({ level: "error", title: "Failed to load conversation", message: (e as Error).message });
        }
    }, [busy, showToast]);

    const onCreateConversation = useCallback(async () => {
        if (busy) return;
        setMessages([]);
        setActiveConvId(null); // a new row will be created on the next send
        setStatusLine(undefined);
        setHistoryOpen(false);
    }, [busy]);

    const onDeleteConversation = useCallback(async (id: string) => {
        if (busy) return;
        const c = conversations.find((x) => x.id === id);
        if (!window.confirm(`Delete conversation "${c?.title ?? id}"?\n\nMessages cannot be recovered.`)) return;
        try {
            await getClient().convDelete(id);
            await refreshConversations();
            if (id === activeConvId) {
                setActiveConvId(null);
                setMessages([]);
            }
        } catch (e) {
            showToast({ level: "error", title: "Delete failed", message: (e as Error).message });
        }
    }, [activeConvId, busy, conversations, refreshConversations, showToast]);

    const onLoad = useCallback(async (m: ModelEntry) => {
        const client = getClient();
        dismissToast("model-load-error");
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
            showToast({
                level: "success",
                title: `Loaded ${m.name}`,
                message: fromCache ? "from OPFS cache" : `downloaded ${fmtBytes(totalBytes)}`,
            });
            // Don't clobber messages here — selecting a conversation
            // before loading the model is a valid flow.
        } catch (e) {
            const err = (e as Error).message ?? String(e);
            setModelStatus("error");
            setStatusText(`load failed: ${err}`);
            setLoadingLabel("");
            showToast({
                id: "model-load-error",
                level: "error",
                title: "Model load failed",
                message: err,
            });
        }
    }, [dismissToast, showToast]);

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
        // This is silent — the displayed history doesn't show it.
        const sysContent = thinking
            ? (systemPrompt.trim() ? `${THINK_TOKEN}${systemPrompt.trim()}` : THINK_TOKEN)
            : systemPrompt.trim();

        // Strip any system messages from local state when building the
        // rendered prompt — the system block is derived from current
        // settings (sysContent above), NOT from whatever was persisted
        // earlier. That way toggling Thinking takes effect on the very
        // next send, even after the chat has started.
        const userTurns = messages.filter((m) => m.role !== "system");
        const history: ChatMessage[] = [
            ...(sysContent ? [{ role: "system" as const, content: sysContent }] : []),
            ...userTurns,
            { role: "user", content: text },
            { role: "model", content: "" },
        ];
        setMessages(history);

        // Ensure we have a conversation row to attach messages to.
        let convId = activeConvId;
        let modelMsgId: string | null = null;
        try {
            if (!convId) {
                const row = await client.convCreate({
                    title: "New chat",
                    model: modelStatus === "ready" ? statusText.split(" ")[0] : null,
                });
                convId = row.id;
                setActiveConvId(convId);
            }
            // Persist user/model turns only. The system message is derived
            // from current settings (Thinking toggle, systemPrompt) on every
            // send — it doesn't belong in long-term storage.
            await client.msgInsert({ conversationId: convId, role: "user", content: text });
            const modelInsert = await client.msgInsert({ conversationId: convId, role: "model", content: "" });
            modelMsgId = modelInsert.messageId;
        } catch (e) {
            // Continue anyway — we'd rather generate without persistence than fail outright.
            showToast({ level: "warn", title: "Persistence failure", message: (e as Error).message });
        }

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
            let pendingDelta = "";
            let lastFlushAt  = performance.now();
            const flushPending = async () => {
                if (!convId || !modelMsgId || pendingDelta.length === 0) return;
                const delta = pendingDelta;
                pendingDelta = "";
                try { await client.msgAppend(convId, modelMsgId, delta); } catch { /* */ }
            };

            for (let i = 0; i < maxTokens; i++) {
                if (cancelRef.current) break;
                if (curIsEos) break;
                const piece = curStr.replaceAll("▁", " ");
                history[history.length - 1].content += piece;
                pendingDelta += piece;
                setMessages([...history]);
                emitted++;

                if ((emitted % 16 === 0) || (performance.now() - lastFlushAt > 750)) {
                    await flushPending();
                    lastFlushAt = performance.now();
                }

                const r = await client.stepAndDecode(next);
                next     = r.next;
                curStr   = r.str ?? "";
                curIsEos = r.isEos;
            }
            await flushPending();
            if (convId) {
                try {
                    await client.convTouch(convId, suggestTitle(text));
                    await client.dbFlush();
                } catch { /* */ }
                void refreshConversations();
            }
            const dt = performance.now() - tg0;
            const tps = emitted > 0 ? (emitted * 1000 / dt) : 0;
            setStatusLine(`pe ${peMs.toFixed(0)} ms · gen ${emitted} tok in ${dt.toFixed(0)} ms · ${tps.toFixed(2)} tok/s`);
            beacon("chat", `gen ${emitted} tok in ${dt.toFixed(0)} ms (${tps.toFixed(2)} tok/s)`);
        } catch (e) {
            const msg = (e as Error).message;
            setStatusLine(`error: ${msg}`);
            if (msg !== "cancelled") {
                showToast({ level: "error", title: "Generation failed", message: msg });
            }
        } finally {
            setBusy(false);
        }
    }, [activeConvId, busy, maxTokens, messages, modelStatus, prompt, refreshConversations, sampling, statusText, systemPrompt, thinking, showToast]);

    const onDeleteModel = useCallback(async (m: ModelEntry) => {
        if (busy) return;
        const sizeLabel = fmtBytes(m.size);
        const ok = window.confirm(
            `Delete cached "${m.name}" from this browser's OPFS?\n\n` +
            `This frees ${sizeLabel} of storage. The model itself stays in ~/.ollama/models — ` +
            `re-loading will download it again.`,
        );
        if (!ok) return;

        const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
        const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";

        const wasLoaded = modelStatus === "ready" && statusText.startsWith(m.name);
        if (wasLoaded) {
            try { await getClient().free(); } catch { /* */ }
            setModelStatus("idle");
            setStatusText("no model");
            setMessages([]);
            setStatusLine(undefined);
        }

        const removed = await wipeModel(modelKey, filename);
        beacon("chat", removed ? `deleted ${m.name} (${sizeLabel})` : `delete ${m.name} no-op (not cached)`);
        if (removed) {
            showToast({ level: "info", title: `Deleted ${m.name}`, message: `Freed ${sizeLabel} from OPFS` });
        } else {
            showToast({ level: "info", title: `No cached copy of ${m.name}` });
        }
    }, [busy, modelStatus, statusText, showToast]);

    const onReset = useCallback(() => {
        if (busy) return;
        setMessages([]);
        setActiveConvId(null);
        setStatusLine(undefined);
        void getClient().reset();
    }, [busy]);

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* Slim toolbar — title + history + settings. */}
            <header className="flex items-center gap-2 border-b border-border bg-card/40 px-3 py-1.5 text-xs safe-top">
                <span className="font-semibold tracking-tight">rullama</span>
                <span
                    className="ml-1 truncate text-xs text-muted-foreground"
                    title={statusText}
                >
                    {modelStatus === "ready" ? statusText : ""}
                </span>
                <div className="ml-auto flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => setHistoryOpen(!historyOpen)}
                        title="Toggle conversation history"
                        aria-pressed={historyOpen}
                    >
                        <History />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => setSettingsOpen(!settingsOpen)}
                        title="Toggle settings"
                        aria-pressed={settingsOpen}
                    >
                        <Settings />
                    </Button>
                </div>
            </header>

            {modelStatus === "loading" && (
                <ModelLoadProgress percent={loadingPercent} label={loadingLabel} />
            )}

            {historyOpen && (
                <ConversationList
                    conversations={conversations}
                    activeId={activeConvId}
                    onSelect={(id) => { void onSelectConversation(id); setHistoryOpen(false); }}
                    onCreate={onCreateConversation}
                    onDelete={(id) => void onDeleteConversation(id)}
                />
            )}

            {settingsOpen && (
                <SettingsDialog
                    modelStatus={modelStatus}
                    loadingPercent={loadingPercent}
                    loadingLabel={loadingLabel}
                    statusText={statusText}
                    onLoadModel={onLoad}
                    onDeleteModel={onDeleteModel}
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
