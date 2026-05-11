import { useCallback, useEffect, useRef, useState } from "react";
import { ModelLoader, ModelLoadProgress, type ModelStatus } from "@/components/ModelLoader";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatPanel } from "@/components/ChatPanel";
import { SettingsDialog, SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { ConversationList } from "@/components/ConversationList";
import { DualSidebarLayout } from "@/components/layouts/DualSidebarLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { type ChatMessage, type SamplingOptions, DEFAULT_SAMPLING, DEFAULT_SYSTEM_PROMPT } from "@/lib/types";
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
    const [conversations, setConversations] = useState<ConversationRow[]>([]);
    const [activeConvId, setActiveConvId]   = useState<string | null>(null);

    // Sidebar visibility — persisted across reloads. Settings defaults
    // open on first ever load so the model picker is visible.
    const [historyOpen,  setHistoryOpen]  = usePersistedState<boolean>("ui.historyOpen",  false);
    const [settingsOpen, setSettingsOpen] = usePersistedState<boolean>("ui.settingsOpen", true);

    // Persisted tunables.
    const [systemPrompt, setSystemPrompt] = usePersistedState<string>("systemPrompt", DEFAULT_SYSTEM_PROMPT);
    const [sampling,     setSampling]     = usePersistedState<SamplingOptions>("sampling", DEFAULT_SAMPLING);
    const [maxTokens,    setMaxTokens]    = usePersistedState<number>("maxTokens", 1024);
    const [thinking,     setThinking]     = usePersistedState<boolean>("thinking", true);

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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Environment probe → sticky toasts on missing capabilities.
    useEffect(() => {
        const has = (k: () => boolean) => { try { return k(); } catch { return false; } };
        if (!has(() => typeof navigator !== "undefined" && "gpu" in navigator)) {
            showToast({
                id: "env-webgpu", level: "error", title: "WebGPU not available",
                message: "rullama cannot run inference without WebGPU. On iOS update to iOS 18+; on desktop use a recent Chrome/Edge/Safari.",
                persist: true,
            });
        }
        if (!has(() => typeof navigator !== "undefined" && !!navigator.storage && typeof navigator.storage.getDirectory === "function")) {
            showToast({
                id: "env-opfs", level: "error", title: "OPFS not available",
                message: "Models larger than ~3 GB need OPFS for streaming. Without it, large GGUFs will OOM the page.",
                persist: true,
            });
        }
        if (!has(() => typeof window !== "undefined" && window.crossOriginIsolated)) {
            showToast({
                id: "env-coi", level: "warn", title: "Cross-origin isolation off",
                message: "Required for SharedArrayBuffer and high-res timers. Make sure the page is served with COOP+COEP headers.",
                persist: true,
            });
        }
    }, [showToast]);

    // Global Escape → stop generation. Mirrors the toolbar Stop button.
    // Attached to window so it fires regardless of focus (input, sidebar
    // toggle, anything). No-op when nothing is generating.
    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape" && busy) {
                cancelRef.current = true;
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [busy]);

    // Bootstrap DB + conversation list on mount.
    useEffect(() => {
        const client = getClient();
        (async () => {
            try {
                await client.dbInit();
                const rows = await client.convList();
                setConversations(rows);
            } catch (e) {
                showToast({ level: "error", title: "Database init failed", message: (e as Error).message });
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

    const onResetDefaults = useCallback(() => {
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
        setSampling(DEFAULT_SAMPLING);
        setMaxTokens(SETTINGS_BOUNDS.maxTokens.fallback);
        setThinking(true);
        showToast({
            level: "success",
            title: "Settings reset to defaults",
        });
    }, [setSystemPrompt, setSampling, setMaxTokens, setThinking, showToast]);

    const onCreateConversation = useCallback(() => {
        if (busy) return;
        setMessages([]);
        setActiveConvId(null);
        setStatusLine(undefined);
        // Clear the worker's KV cache up front. onSend would do this on
        // the next send anyway, but releasing the memory immediately is
        // friendlier when the user is intentionally starting fresh.
        void getClient().reset();
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
            const url = blobUrl(m);

            const t0 = performance.now();
            // The writer worker emits a progress event per stream chunk
            // (~hundreds per second on a fast link). Updating React state
            // that often makes the label flicker; throttle to ~4 Hz, but
            // always emit the final tick so 100 % shows the real number.
            let lastLabelAt = 0;
            const { totalBytes, fromCache } = await ensureModel(url, modelKey, filename, ({ bytesWritten, totalBytes }) => {
                if (totalBytes > 0) {
                    setLoadingPercent((bytesWritten / totalBytes) * 100);
                    const now     = performance.now();
                    const done    = bytesWritten >= totalBytes;
                    if (done || now - lastLabelAt > 250) {
                        lastLabelAt = now;
                        const elapsed = (now - t0) / 1000;
                        const rate    = bytesWritten / Math.max(elapsed, 0.001);
                        setLoadingLabel(`${fmtBytes(bytesWritten)} / ${fmtBytes(totalBytes)} — ${fmtBytes(rate)}/s`);
                    }
                } else {
                    const now = performance.now();
                    if (now - lastLabelAt > 250) {
                        lastLabelAt = now;
                        setLoadingLabel(fmtBytes(bytesWritten));
                    }
                }
            });
            if (fromCache) {
                beacon("chat", `OPFS cache hit (${fmtBytes(totalBytes)})`);
            } else {
                beacon("chat", `downloaded ${fmtBytes(totalBytes)} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
            }

            setLoadingLabel("loading into wasm…");
            const mobile = isMobileUA();
            // Bumped 512 → 2048. The 512 was a conservative crash-avoidance
            // number from the iPhone-load-path debugging phase; once the
            // worker + sync-OPFS + per-tile range-fetch combination
            // stabilised, we've never come close to the actual GPU memory
            // budget. A18 exposes max_buffer_size = 1 GiB, KV-per-token is
            // ~70 KB across all layers, so 2048 ctx ≈ 144 MB of GPU
            // memory — comfortable headroom on a phone.
            const mobileMaxCtx = 2048;
            // textOnly is forced when:
            //   - mobile (no headroom for vision/audio towers), OR
            //   - the remote URL points at a text-only blob (HF-style:
            //     `mmproj` ships separately and audio isn't in GGUF at
            //     all). R2-hosted Ollama-style blobs have `multimodal:
            //     true` and load the full weight set on desktop.
            const textOnlyRemote = !!m.url && !m.multimodal;
            await client.load(modelKey, filename, {
                maxContext: mobile ? mobileMaxCtx : 0,
                textOnly:   mobile || textOnlyRemote,
            });
            setModelStatus("ready");
            setStatusText(`${m.name}${fromCache ? " ⚡" : ""}`);
            setLoadingLabel("");
            showToast({
                level: "success", title: `Loaded ${m.name}`,
                message: fromCache ? "from OPFS cache" : `downloaded ${fmtBytes(totalBytes)}`,
            });
        } catch (e) {
            const err = (e as Error).message ?? String(e);
            setModelStatus("error");
            setStatusText(`load failed: ${err}`);
            setLoadingLabel("");
            showToast({
                id: "model-load-error", level: "error",
                title: "Model load failed", message: err,
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

        const sysContent = thinking
            ? (systemPrompt.trim() ? `${THINK_TOKEN}${systemPrompt.trim()}` : THINK_TOKEN)
            : systemPrompt.trim();

        // Two parallel histories: `displayHistory` for the chat UI (no
        // system message — it's a control signal, not user content) and
        // `renderHistory` for tokenization (with system at the front).
        const userTurns = messages.filter((m) => m.role !== "system");
        const displayHistory: ChatMessage[] = [
            ...userTurns,
            { role: "user", content: text },
            { role: "model", content: "" },
        ];
        setMessages(displayHistory);
        const renderHistory: ChatMessage[] = sysContent
            ? [{ role: "system", content: sysContent }, ...displayHistory.slice(0, -1)]
            : displayHistory.slice(0, -1);

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
            await client.msgInsert({ conversationId: convId, role: "user", content: text });
            const modelInsert = await client.msgInsert({ conversationId: convId, role: "model", content: "" });
            modelMsgId = modelInsert.messageId;
        } catch (e) {
            showToast({ level: "warn", title: "Persistence failure", message: (e as Error).message });
        }

        try {
            await client.setSampling(sampling);
            await client.reset();
            const rendered = await client.renderChat(renderHistory, false);
            const ids = await client.encode(rendered);

            const t0 = performance.now();
            let next = 0;
            for (let i = 0; i < ids.length; i++) {
                if (cancelRef.current) throw new Error("cancelled");
                next = await client.step(ids[i]);
            }
            const peMs = performance.now() - t0;

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
                displayHistory[displayHistory.length - 1].content += piece;
                pendingDelta += piece;
                setMessages([...displayHistory]);
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
            `Delete cached "${m.name}" (${sizeLabel})?\n\n` +
            `This only removes the local copy in this browser. ` +
            `Loading again will re-download it.`,
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

    // Active conversation title for the header.
    const activeTitle = activeConvId
        ? conversations.find((c) => c.id === activeConvId)?.title
        : undefined;

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* ─── top header (3rem / 48px tall — matches DualSidebarLayout offset) ─── */}
            {/* `min-h-12` not `h-12` so the safe-area-inset-top padding
                actually grows the header on iPhones with a notch /
                Dynamic Island — fixed h-12 was stuffing all content
                under the status bar in standalone PWA mode. */}
            <header className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-card/50 px-3 safe-top">
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setHistoryOpen(!historyOpen)}
                    title="Toggle conversation history"
                    aria-pressed={historyOpen}
                >
                    <History />
                </Button>
                <span className="font-semibold tracking-tight">rullama</span>
                {activeTitle && (
                    <span className="hidden truncate text-xs text-muted-foreground sm:inline">
                        / {activeTitle}
                    </span>
                )}
                <div className="ml-auto flex items-center gap-2">
                    {modelStatus === "ready" && (
                        <Badge
                            tone="ok"
                            className="hidden max-w-[14rem] truncate sm:inline-flex"
                            title={statusText}
                        >
                            {statusText}
                        </Badge>
                    )}
                    {modelStatus === "loading" && (
                        <Badge tone="warn" className="max-w-[14rem] truncate">
                            {loadingLabel || "loading…"}
                        </Badge>
                    )}
                    {modelStatus === "error" && (
                        <Badge tone="err">error</Badge>
                    )}
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
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

            <DualSidebarLayout
                leftOpen={historyOpen}
                rightOpen={settingsOpen}
                onToggleLeft={setHistoryOpen}
                onToggleRight={setSettingsOpen}
                leftWidth={280}
                rightWidth={320}
                leftSidebar={
                    <ConversationList
                        conversations={conversations}
                        activeId={activeConvId}
                        onSelect={(id) => { void onSelectConversation(id); }}
                        onCreate={onCreateConversation}
                        onDelete={(id) => void onDeleteConversation(id)}
                    />
                }
                rightSidebar={
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
                        onResetDefaults={onResetDefaults}
                    />
                }
            >
                <ChatPanel
                    messages={messages}
                    emptyState={
                        modelStatus !== "ready" ? (
                            <div className="mx-auto mt-6 w-full max-w-md">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Load a model</CardTitle>
                                        <CardDescription>
                                            Pick a model and click Load. It'll cache to this
                                            browser's local OPFS storage so subsequent visits
                                            are instant. Past conversations live in the
                                            History sidebar.
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <ModelLoader
                                            status={modelStatus}
                                            loadingPercent={loadingPercent}
                                            loadingLabel={loadingLabel}
                                            statusText={statusText}
                                            onLoad={onLoad}
                                            onDelete={onDeleteModel}
                                        />
                                    </CardContent>
                                </Card>
                            </div>
                        ) : (
                            <WelcomeScreen
                                modelName={statusText}
                                onSuggest={(p) => setPrompt(p)}
                            />
                        )
                    }
                    canType={modelStatus === "ready"}
                    canSend={modelStatus === "ready" && !busy && prompt.trim().length > 0}
                    canStop={busy}
                    canNewChat={modelStatus === "ready" && messages.length > 0 && !busy}
                    prompt={prompt}
                    onPromptChange={setPrompt}
                    onSend={onSend}
                    onStop={() => { cancelRef.current = true; }}
                    onNewChat={onCreateConversation}
                    statusLine={statusLine}
                />
            </DualSidebarLayout>
        </div>
    );
}
