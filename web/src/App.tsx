import { useCallback, useEffect, useRef, useState } from "react";
import { ModelLoader, ModelLoadProgress, LoadingModel } from "@/components/ModelLoader";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatPanel } from "@/components/ChatPanel";
import { RestartOverlay } from "@/components/RestartOverlay";
import { SettingsDialog } from "@/components/SettingsDialog";
import { VoicePanel } from "@/components/VoicePanel";
import { ConversationList } from "@/components/ConversationList";
import { DualSidebarLayout } from "@/components/layouts/DualSidebarLayout";
import { getClient, teardownInferenceCore } from "@/lib/inference";
import { disposeSharedClone } from "@/lib/clone-client";
import { disposeSharedTts } from "@/lib/tts-client";
import { ChatSettings } from "@/components/ChatSettings";
import { AppHeader } from "@/components/AppHeader";
import { KnowledgeTab } from "@/components/KnowledgeTab";
import { ImageTab } from "@/components/ImageTab";
import { TrainingOverlay } from "@/components/TrainingOverlay";
import { UnsupportedScreen } from "@/components/UnsupportedScreen";
import { useDeviceTier } from "@/lib/capability";
import { usePersistedState } from "@/lib/persisted";
import { useIOSKeyboard } from "@/lib/useIOSKeyboard";
import { useWakeLock } from "@/lib/wakeLock";
import { useTrainingCapability } from "@/components/FineTunePanel";
import { UpdateBanner } from "@/components/UpdateBanner";
import { ApplyingOverlay } from "@/components/ApplyingOverlay";
import { useChatTunables } from "@/hooks/useChatTunables";
import { useToolSettings } from "@/hooks/useToolSettings";
import { useSidebars } from "@/hooks/useSidebars";
import { usePwaUpdate } from "@/hooks/usePwaUpdate";
import { useWaitInfo } from "@/hooks/useWaitInfo";
import { useModelLoad } from "@/hooks/useModelLoad";
import { useChatEngine } from "@/hooks/useChatEngine";
import { useSessionLifecycle } from "@/hooks/useSessionLifecycle";
import { useCrossTabSync } from "@/hooks/useCrossTabSync";


export function App() {
    // Training capability — drives Fine-tune tab gating + the
    // "training not supported" screen. Detected once on mount via
    // `useTrainingCapability()` (WebGPU + min GPU buffer + min system
    // RAM + non-iOS UA).
    const trainingCap = useTrainingCapability();

    // View routing — Chat / Voice / Settings. Persisted so a reload doesn't
    // bounce the user out of the tab they were in. (Fine-tune is no longer a
    // tab — it's a full-screen overlay launched from Chat's sidebar; voice
    // learning likewise from Voice's sidebar. Keeping training in the same
    // engine context as its tab is what removed the cross-engine reload that
    // used to fail with "model load failed".)
    const [view, setView] = usePersistedState<"chat" | "voice" | "knowledge" | "image" | "settings">("rullama:view", "chat");
    // Migrate any persisted legacy "finetune" view to "chat".
    useEffect(() => {
        if ((view as string) === "finetune") setView("chat");
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    // Full-screen training overlay: which one is open (null = none).
    const [training, setTraining] = useState<null | "finetune" | "voicelearn">(null);
    // Device-capability tier (gates the boot, engine co-residency, model markers).
    const { tier, probe } = useDeviceTier();
    const premium = tier === "premium";
    // **D3 — chat-during-training gate.** When a Fine-tune run is
    // active in this (or any other) tab, the Model is owned by the
    // training session and chat-side step RPCs would fail with
    // "model is owned by an active training session". Reflect that
    // in the UI so the user sees a clear "training in progress"
    // affordance instead of an opaque error toast on Send. Sourced
    // from the worker's `trainingStarted` / `trainingFinished`
    // notifies, which fan out across tabs via the SharedWorker
    // router.
    const [trainingInProgress, setTrainingInProgress] = useState<boolean>(false);

    // Adapter currently active in chat. Persisted across reloads so a
    // page refresh doesn't silently drop a trained adapter — the bytes
    // are in OPFS regardless, but the "this one is loaded into Model"
    // state used to reset to null on boot, forcing the user to
    // re-apply manually via the Fine-tune tab. The restore-on-boot
    // effect below re-applies the saved adapter once the model is
    // ready; failure clears the persisted name so we don't keep
    // trying.
    const [activeAdapter, setActiveAdapter] = usePersistedState<string | null>("activeAdapter", null);

    // PWA update banner state + apply/dismiss handlers + the boot-time
    // version check (see usePwaUpdate). The cross-tab `meta` /
    // `updateAvailable` / `applyingUpdate` notifies from the SharedWorker
    // router (useCrossTabSync) also drive this via the returned setters.
    // The banner is gated on `!busy` at render (defer mid-generation);
    // the apply overlay renders whenever the version mismatch is applied.
    const {
        updateVersion, setUpdateVersion,
        applyingUpdate, setApplyingUpdate,
        onApplyUpdate, onDismissUpdate,
    } = usePwaUpdate();

    // Wait-reason coordination for the model loader (see useWaitInfo).
    // useCrossTabSync pushes the most-recent wait reason via setWaitInfo;
    // render passes `waitInfo?.message ?? loadingLabel` so a fresh wait
    // label supersedes the normal progress label.
    const { waitInfo, setWaitInfo } = useWaitInfo();

    // Tool calling + Tools-tab settings (schema injection, weather
    // executor, GPS) — see useToolSettings.
    const {
        toolMode, setToolMode,
        weatherApiKey, setWeatherApiKey,
        weatherUnits, setWeatherUnits,
        useGps, setUseGps,
    } = useToolSettings();

    // Sidebar visibility (persisted per sidebar, defaulting open when
    // docked) + the portal-host elements — see useSidebars.
    const {
        historyOpen, setHistoryOpen,
        chatSettingsOpen, setChatSettingsOpen,
        voiceTabSettingsOpen, setVoiceTabSettingsOpen,
        voiceTabSettingsEl, setVoiceTabSettingsEl,
        voiceClipsOpen, setVoiceClipsOpen,
        voiceClipsEl, setVoiceClipsEl,
        fineTuneSettingsOpen, setFineTuneSettingsOpen,
        fineTuneSettingsEl, setFineTuneSettingsEl,
    } = useSidebars();

    // Persisted chat/voice tunables (system prompt, sampling, max tokens,
    // thinking, voice) + their bounds-sanitization + reset — see
    // useChatTunables.
    const {
        systemPrompt, setSystemPrompt,
        sampling, setSampling,
        maxTokens, setMaxTokens,
        thinking, setThinking,
        voice, setVoice,
        onResetDefaults,
    } = useChatTunables();

    // ── model ↔ chat cycle break ──────────────────────────────────────
    // The model loader (eject/delete) must clear chat display state and
    // no-op mid-generation; the chat engine must read model status. To
    // avoid a creation-order cycle, the loader reaches chat through these
    // refs, kept current by the effect just below.
    const busyRef = useRef(false);
    const resetChatRef = useRef<() => void>(() => {});
    // The model loader's "preparing" phase pre-warms the system prompt, but
    // that lives in the chat engine (created below). Reach it through a ref,
    // kept current by the effect that maintains the other cycle-break refs.
    const prepareRef = useRef<((report: (p: number, l: string) => void) => Promise<void>) | null>(null);
    // De-dup the system pre-warm: `sysWarmSigRef` is the current warm-input
    // signature; `warmedSigRef` is the signature last successfully warmed.
    // Lets the belt-and-suspenders re-warm effect skip when nothing changed.
    const sysWarmSigRef = useRef("");
    const warmedSigRef = useRef("");

    // Model-load lifecycle (load / eject / delete / cancel, auto-load,
    // reload-after-engine-swap, boot splash). Owns the model status
    // surface; its setters feed the cross-tab sync.
    const {
        modelStatus, setModelStatus,
        loadedIsDiffusion,
        statusText, setStatusText,
        hasVision, setHasVision,
        hasAudio, setHasAudio,
        setMetaInit,
        selectedModelName, setSelectedModelName,
        loadingPercent, loadingLabel,
        lastLoadedDigest,
        onLoad, onCancelDownload, onCancelAndDeleteDownload,
        reloadInferenceModel, onDeleteModel, onEjectModel,
    } = useModelLoad({
        tier,
        waitInfo,
        onUnloadChat: () => resetChatRef.current(),
        isBusy: () => busyRef.current,
        onPrepare: async (report) => {
            await (prepareRef.current?.(report) ?? Promise.resolve());
            // Record what we just warmed so the belt re-warm effect doesn't
            // redundantly redo it on the same config.
            warmedSigRef.current = sysWarmSigRef.current;
        },
    });

    // Chat engine — conversations, messages, attachments, RAG, and the
    // generation + suspend/resume machinery. Reads model/tunable/tool
    // state via params.
    const {
        messages, prompt, setPrompt, busy, statusLine, visionEncodeState,
        runningConvIds, queuedConvIds, activeConvIsGenerating, activeConvIsQueued,
        pendingImages, pendingAudio, conversations, setConversations, activeConvId,
        inflightRef,
        onSelectConversation, onCreateConversation, onDeleteConversation,
        onAttachFiles, onRemoveImage, onCaptureAudio, onRemoveAudio, onAudioError,
        onSend, onStop, resetForUnload, warmSystemPrompt,
    } = useChatEngine({
        modelStatus, loadedIsDiffusion, statusText, lastLoadedDigest,
        hasVision, hasAudio, systemPrompt, sampling, maxTokens, thinking,
        toolMode, weatherApiKey, weatherUnits, useGps, activeAdapter,
    });

    // Keep the cycle-break refs current for the model loader.
    useEffect(() => {
        busyRef.current = busy;
        resetChatRef.current = resetForUnload;
        prepareRef.current = warmSystemPrompt;
    });

    // Signature of everything that determines the warmed system block.
    const sysWarmSig = JSON.stringify({ systemPrompt, thinking, toolMode, activeAdapter });
    sysWarmSigRef.current = sysWarmSig;

    // **Belt-and-suspenders system pre-warm.** The loader's "preparing"
    // phase warms on load, but the warmed prefix can go stale (thinking /
    // tool-mode toggled, adapter applied/cleared) or never have taken. When
    // the model is ready and the warm signature differs from what we last
    // warmed, re-warm so new chats keep hot-starting (kvReusePlan reuses the
    // resident system prefix). Deduped + debounced; skipped while busy.
    useEffect(() => {
        if (modelStatus !== "ready" || loadedIsDiffusion || busy) return;
        if (warmedSigRef.current === sysWarmSig) return;
        const t = setTimeout(() => {
            void (async () => {
                try { await warmSystemPrompt(); warmedSigRef.current = sysWarmSig; }
                catch { /* non-fatal — first chat just prefills once */ }
            })();
        }, 350);
        return () => clearTimeout(t);
    }, [sysWarmSig, modelStatus, loadedIsDiffusion, busy, warmSystemPrompt]);

    // Save a new system prompt, then re-warm the KV cache with it so the
    // next new chat hot-starts (and the current conversation's next turn
    // reuses the new system block). The editor awaits this to show its
    // "preparing…" state. Best-effort warm — a failure just means the
    // first chat prefills the new system once.
    const onSaveSystemPrompt = useCallback(async (value: string) => {
        setSystemPrompt(value);
        if (modelStatus === "ready") {
            try { await warmSystemPrompt(undefined, { systemPrompt: value }); }
            catch { /* non-fatal */ }
        }
    }, [setSystemPrompt, warmSystemPrompt, modelStatus]);

    // Page/session lifecycle: env probe, crash-detect, pagehide marker.
    useSessionLifecycle(setView);

    // Cross-tab state sync via the SharedWorker. Setters/refs are sourced
    // from their owning hooks; the effect itself lives in the hook.
    useCrossTabSync({
        setMetaInit, setModelStatus, setHasVision, setHasAudio, setStatusText,
        setUpdateVersion, setApplyingUpdate, setConversations, setActiveAdapter,
        setTrainingInProgress, setWaitInfo, inflightRef,
    });

    // iOS keyboard handling — snaps the visual viewport back to the top
    // when the keyboard dismisses, so the page doesn't end up offset
    // a few px above the layout viewport (a classic iOS-Safari quirk).
    useIOSKeyboard(true);

    // Hold the screen awake during the two long-running operations a
    // user actually waits on — model download/load and token generation.
    // No-op on platforms without `navigator.wakeLock` (older iOS, private
    // mode). See `lib/wakeLock.ts` for the iOS hide-release dance.
    useWakeLock(modelStatus === "loading" || modelStatus === "preparing" || busy);

    // **D1 — restore active adapter on reload.** The adapter bytes
    // survive in OPFS but the "this one is loaded into Model" state
    // used to reset to null on every boot, leaving the user to manually
    // re-apply via the Fine-tune tab. Once the model is ready and an
    // activeAdapter is persisted but not yet applied to the worker,
    // re-apply it. A single retry on the next-`modelStatus` change is
    // enough; failure clears the persisted name so we don't keep
    // trying for a missing/incompatible adapter.
    const adapterRestoredRef = useRef(false);
    useEffect(() => {
        if (adapterRestoredRef.current) return;
        if (modelStatus !== "ready") return;
        if (!activeAdapter) return;
        adapterRestoredRef.current = true;
        (async () => {
            const c = getClient();
            try {
                const list = await c.trainingListAdapters();
                if (list.active === activeAdapter) return; // already applied
                if (!list.entries.some((e) => e.name === activeAdapter)) {
                    console.warn(`[rullama] persisted activeAdapter '${activeAdapter}' is no longer in the library — clearing`);
                    setActiveAdapter(null);
                    return;
                }
                // trainingApplyAdapter is session-gated. Wrap in
                // withSession so the apply queues behind any other
                // tab's active work and releases cleanly when done.
                await c.withSession(() => c.trainingApplyAdapter(activeAdapter));
                // Applying the adapter invalidated the warmed system KV
                // (its K/V was computed under base weights). Re-warm with the
                // adapter active so the first chat still hot-starts.
                warmedSigRef.current = "";
                try { await warmSystemPrompt(); warmedSigRef.current = sysWarmSigRef.current; }
                catch { /* non-fatal */ }
            } catch (e) {
                console.warn(`[rullama] failed to restore activeAdapter '${activeAdapter}' on boot:`, e);
                setActiveAdapter(null);
            }
        })();
    }, [modelStatus, activeAdapter, setActiveAdapter, warmSystemPrompt]);

    // **One engine resident at a time (except Premium).** Inference (Gemma) and the TTS/clone
    // engines can't share a phone GPU, and on iOS WebKit only a worker's DEATH reclaims its GPU
    // memory (model.free() and GPUBuffer.destroy() don't, observably — bug 302711 family). So on a
    // Chat↔Voice tab change, tear down the engine we're NOT using and (re)load the one we are —
    // replicating the user's working "fresh reload" automatically. Training overlays don't change
    // `view`, so launching Fine-tune (over Chat) / Voice-learning (over Voice) performs NO swap;
    // training shares its tab's engine. **Premium tier short-circuits the teardown** so inference +
    // TTS stay co-resident (the user has the VRAM; Chat↔Voice is then instant). `settings` needs
    // neither engine, so it no-ops — switching to Settings never loads or unloads a model.
    const engineSwapMounted = useRef(false);
    useEffect(() => {
        const firstRun = !engineSwapMounted.current;
        engineSwapMounted.current = true;
        if (premium) return; // keep both engines warm on high-VRAM machines
        const needsInference = view === "chat";
        const needsVoice = view === "voice";
        // The image engine (Z-Image-Turbo) is GPU-heavy and lives in its own
        // tab; treat it like Voice — don't keep the chat core resident
        // alongside it. The ImageTab owns its own load/unload (client.image.*),
        // so we only tear the chat core down here, same as the voice path.
        const needsImage = view === "image";
        if ((needsVoice || needsImage) && !needsInference) {
            teardownInferenceCore();
            setModelStatus("idle"); // returning to chat re-triggers the load below
        } else if (needsInference && !needsVoice && !needsImage) {
            disposeSharedClone();
            disposeSharedTts();
            if (!firstRun && modelStatus !== "ready" && modelStatus !== "loading") {
                void reloadInferenceModel();
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [view, premium]);

    // Esc closes the training overlay.
    useEffect(() => {
        if (!training) return;
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setTraining(null); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [training]);

    // Active conversation title for the header.
    const activeTitle = activeConvId
        ? conversations.find((c) => c.id === activeConvId)?.title
        : undefined;

    // Hard block below the minimum spec — render a clean screen and DON'T
    // boot the engine (the auto-load effect is gated on tier too). This is
    // what stops incapable devices (e.g. iPhone 7, no WebGPU) from boot-
    // looping the heavy app. `tier === null` means the async probe is still
    // in flight; render the normal shell (the engine stays gated until it
    // resolves).
    if (tier === "unsupported") return <UnsupportedScreen probe={probe} />;

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* PWA update banner. Detected via boot-time /version.json
                fetch (lib/version.ts) and broadcast across tabs via
                the SharedWorker. Deferred while mid-generation so an
                in-flight chat reply is never interrupted. */}
            {updateVersion && !busy && !applyingUpdate && (
                <UpdateBanner
                    version={updateVersion}
                    onApply={onApplyUpdate}
                    onDismiss={onDismissUpdate}
                />
            )}
            {/* ─── top header (3rem / 48px tall — matches DualSidebarLayout offset) ─── */}
            <AppHeader
                view={view}
                onSelectView={setView}
                historyOpen={historyOpen}
                onToggleHistory={() => setHistoryOpen(!historyOpen)}
                activeTitle={activeTitle}
                activeAdapter={activeAdapter}
            />

            {(modelStatus === "loading" || modelStatus === "preparing") && (
                <ModelLoadProgress percent={loadingPercent} label={waitInfo?.message ?? loadingLabel} />
            )}

            <DualSidebarLayout
                leftOpen={
                    view === "chat" ? historyOpen
                    : view === "voice" ? voiceClipsOpen
                    : false
                }
                onToggleLeft={view === "chat" ? setHistoryOpen : setVoiceClipsOpen}
                leftWidth={280}
                // Left sidebar: chat → conversation list; voice → generated-clips
                // list (portaled from VoicePanel). Right sidebar is each tab's
                // settings. Each gets its own chevron toggle.
                leftSidebar={
                    view === "chat" ? (
                        <ConversationList
                            conversations={conversations}
                            activeId={activeConvId}
                            runningConvIds={runningConvIds}
                            queuedConvIds={queuedConvIds}
                            onSelect={(id) => { void onSelectConversation(id); }}
                            onCreate={() => void onCreateConversation()}
                            onDelete={(id) => void onDeleteConversation(id)}
                        />
                    ) : view === "voice" ? (
                        <div ref={setVoiceClipsEl} className="h-full" />
                    ) : undefined
                }
                rightOpen={
                    view === "chat" ? chatSettingsOpen
                    : view === "voice" ? voiceTabSettingsOpen
                    : false
                }
                onToggleRight={
                    view === "chat" ? setChatSettingsOpen
                    : setVoiceTabSettingsOpen
                }
                rightWidth={340}
                rightSidebar={
                    view === "chat" ? (
                        // Per-tab chat settings: the Gemma model block + Fine-tune launcher,
                        // plus system prompt / sampling / thinking. Logs / app-data / High-VRAM
                        // toggle stay in the global Settings view.
                        <ChatSettings
                            modelStatus={modelStatus}
                            loadingPercent={loadingPercent}
                            loadingLabel={waitInfo?.message ?? loadingLabel}
                            statusText={statusText}
                            onLoadModel={onLoad}
                            onDeleteModel={onDeleteModel}
                            onEjectModel={onEjectModel}
                            selectedModelName={selectedModelName}
                            onSelectModel={setSelectedModelName}
                            preferredDigest={lastLoadedDigest}
                            onCancelDownload={onCancelDownload}
                            onCancelAndDeleteDownload={onCancelAndDeleteDownload}
                            onOpenFineTune={() => setTraining("finetune")}
                            canFineTune={modelStatus === "ready" && trainingCap.status === "ok"}
                            fineTuneReason={
                                modelStatus !== "ready" ? "Load a model first"
                                : trainingCap.status === "checking" ? "Checking device capability…"
                                : trainingCap.status === "blocked" ? trainingCap.title
                                : "Fine-tuning unavailable on this device"
                            }
                            systemPrompt={systemPrompt}
                            onSaveSystemPrompt={onSaveSystemPrompt}
                            sampling={sampling}
                            onSamplingChange={setSampling}
                            maxTokens={maxTokens}
                            onMaxTokensChange={setMaxTokens}
                            thinking={thinking}
                            onThinkingChange={setThinking}
                            toolMode={toolMode}
                            onToolModeChange={setToolMode}
                            weatherApiKey={weatherApiKey}
                            onWeatherApiKeyChange={setWeatherApiKey}
                            weatherUnits={weatherUnits}
                            onWeatherUnitsChange={setWeatherUnits}
                            useGps={useGps}
                            onUseGpsChange={setUseGps}
                            onResetDefaults={onResetDefaults}
                            voice={voice}
                            onVoiceChange={setVoice}
                            canRecord={modelStatus === "ready" && hasAudio}
                        />
                    ) : view === "voice" ? (
                        // VoicePanel portals its voice picker / clone-model block into this host.
                        <div ref={setVoiceTabSettingsEl} className="h-full" />
                    ) : undefined
                }
            >
                {view === "voice" ? (
                    <VoicePanel settingsHostEl={voiceTabSettingsEl} clipsHostEl={voiceClipsEl} onOpenVoiceLearn={() => setTraining("voicelearn")} />
                ) : view === "knowledge" ? (
                    <KnowledgeTab activeConvId={activeConvId} />
                ) : view === "image" ? (
                    <ImageTab />
                ) : view === "settings" ? (
                    // Centered max-width wrapper so the form controls
                    // don't stretch across the full main-content width
                    // on desktop monitors. `h-full min-h-0` keeps the
                    // height-fill contract SettingsDialog expects from
                    // its parent (it uses h-full internally).
                    <div className="mx-auto flex h-full w-full max-w-3xl min-h-0 flex-col">
                        <SettingsDialog />
                    </div>
                ) : (
                <ChatPanel
                    messages={messages}
                    emptyState={
                        modelStatus === "ready" ? (
                            <WelcomeScreen
                                modelName={statusText}
                                onSuggest={(p) => setPrompt(p)}
                            />
                        ) : (modelStatus === "loading" || modelStatus === "preparing") ? (
                            <div className="mx-auto mt-6 w-full max-w-md">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Loading model</CardTitle>
                                        <CardDescription>
                                            Setting up the model in your browser — this runs
                                            entirely on-device. The first load caches to local
                                            OPFS storage so future visits are instant.
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <LoadingModel
                                            status={modelStatus}
                                            percent={loadingPercent}
                                            label={waitInfo?.message ?? loadingLabel}
                                            modelName={selectedModelName || statusText}
                                            onCancel={onCancelDownload}
                                            onCancelDelete={onCancelAndDeleteDownload}
                                        />
                                    </CardContent>
                                </Card>
                            </div>
                        ) : (
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
                                            loadingLabel={waitInfo?.message ?? loadingLabel}
                                            statusText={statusText}
                                            onLoad={onLoad}
                                            onDelete={onDeleteModel}
                                            onCancel={onCancelDownload}
                                            onCancelDelete={onCancelAndDeleteDownload}
                                            selected={selectedModelName}
                                            onSelect={setSelectedModelName}
                                            preferredDigest={lastLoadedDigest}
                                        />
                                    </CardContent>
                                </Card>
                            </div>
                        )
                    }
                    canType={modelStatus === "ready" && !trainingInProgress}
                    canSend={
                        // No longer gated on generation: sending while a turn
                        // is in flight QUEUES the message (it runs serially
                        // after) rather than blocking.
                        modelStatus === "ready"
                        && !trainingInProgress
                        && (prompt.trim().length > 0 || pendingImages.length > 0 || pendingAudio.length > 0)
                    }
                    // Stop targets the conversation on screen: show it when the
                    // active conversation has a running or queued job (Stop
                    // cancels the run / drops the queued message). Other
                    // conversations keep generating untouched.
                    canStop={activeConvIsGenerating || activeConvIsQueued}
                    canAttach={modelStatus === "ready" && (hasVision || hasAudio) && !trainingInProgress}
                    canRecord={modelStatus === "ready" && hasAudio && !trainingInProgress}
                    // Speak-a-reply runs the Kokoro TTS engine alongside inference — needs at least
                    // the recommended GPU tier. Hidden on the minimum (mobile) tier.
                    canSpeak={tier === "desktop" || tier === "premium"}
                    statusLine={trainingInProgress
                        ? "Training session is active — open Fine-tune and Save / Apply / Discard the adapter to return to chat."
                        : statusLine}
                    pendingImages={pendingImages}
                    pendingAudio={pendingAudio.map((a) => ({ durationMs: a.durationMs }))}
                    voice={voice}
                    prompt={prompt}
                    onPromptChange={setPrompt}
                    onSend={onSend}
                    onStop={onStop}
                    onAttachFiles={(files) => { void onAttachFiles(files); }}
                    onRemoveImage={onRemoveImage}
                    onCaptureAudio={onCaptureAudio}
                    onRemoveAudio={onRemoveAudio}
                    onAudioError={onAudioError}
                    pipelineProgress={visionEncodeState}
                />
                )}
            </DualSidebarLayout>

            {/* Full-screen training overlay. Launched from a tab's sidebar button;
                Chat/Voice stay mounted underneath so the engine stays GPU-resident
                (no swap — training shares its tab's engine). */}
            {training && (
                <TrainingOverlay
                    training={training}
                    onClose={() => setTraining(null)}
                    trainingCap={trainingCap}
                    fineTuneSettingsOpen={fineTuneSettingsOpen}
                    onToggleFineTuneSettings={setFineTuneSettingsOpen}
                    fineTuneSettingsEl={fineTuneSettingsEl}
                    setFineTuneSettingsEl={setFineTuneSettingsEl}
                    modelStatus={modelStatus}
                    activeAdapter={activeAdapter}
                    onAdapterChanged={setActiveAdapter}
                />
            )}

            <RestartOverlay />
            {applyingUpdate && updateVersion && <ApplyingOverlay version={updateVersion} />}
        </div>
    );
}
