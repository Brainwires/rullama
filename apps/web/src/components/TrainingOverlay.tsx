/// Full-screen training overlay, launched from a tab's sidebar button. Chat/Voice
/// stay mounted underneath so the engine stays GPU-resident (no swap — training
/// shares its tab's engine). "finetune" mounts FineTunePanel (with its
/// hyperparameter column portaled into a right sidebar); "voicelearn" mounts
/// VoiceTrainPanel. Extracted from App.tsx.

import { Sparkles, AudioLines, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DualSidebarLayout } from "@/components/layouts/DualSidebarLayout";
import { FineTunePanel, TrainingBlockedScreen, type TrainingCapability } from "@/components/FineTunePanel";
import { VoiceTrainPanel } from "@/components/VoiceTrainPanel";
import type { ModelStatus } from "@/components/ModelLoader";

interface TrainingOverlayProps {
    training: "finetune" | "voicelearn";
    onClose: () => void;
    trainingCap: TrainingCapability;
    fineTuneSettingsOpen: boolean;
    onToggleFineTuneSettings: (next: boolean) => void;
    fineTuneSettingsEl: HTMLDivElement | null;
    setFineTuneSettingsEl: (el: HTMLDivElement | null) => void;
    modelStatus: ModelStatus;
    activeAdapter: string | null;
    onAdapterChanged: (name: string | null) => void;
}

export function TrainingOverlay({
    training,
    onClose,
    trainingCap,
    fineTuneSettingsOpen,
    onToggleFineTuneSettings,
    fineTuneSettingsEl,
    setFineTuneSettingsEl,
    modelStatus,
    activeAdapter,
    onAdapterChanged,
}: TrainingOverlayProps) {
    return (
        <div className="fixed inset-0 z-50 flex flex-col bg-background">
            <header className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-card/50 px-3 safe-top">
                {training === "finetune"
                    ? <Sparkles className="size-4 text-muted-foreground" />
                    : <AudioLines className="size-4 text-muted-foreground" />}
                <span className="text-sm font-medium">
                    {training === "finetune" ? "Fine-tune" : "Voice learning"}
                </span>
                <Button
                    variant="ghost"
                    size="sm"
                    className="ml-auto h-8 gap-1 text-xs"
                    onClick={onClose}
                    title="Close (Esc)"
                >
                    <X className="size-4" /> Close
                </Button>
            </header>
            {training === "finetune" && trainingCap.status === "ok" ? (
                // FineTunePanel's hyperparameter column lives in the right
                // sidebar (portaled via settingsHostEl), with its own chevron
                // toggle — same pattern as the Chat/Voice tabs.
                <DualSidebarLayout
                    rightOpen={fineTuneSettingsOpen}
                    onToggleRight={onToggleFineTuneSettings}
                    rightWidth={340}
                    rightSidebar={<div ref={setFineTuneSettingsEl} className="h-full" />}
                >
                    <FineTunePanel
                        modelStatus={modelStatus}
                        activeAdapter={activeAdapter}
                        onAdapterChanged={onAdapterChanged}
                        settingsHostEl={fineTuneSettingsEl}
                    />
                </DualSidebarLayout>
            ) : (
                <div className="min-h-0 flex-1 overflow-hidden">
                    {training === "voicelearn" ? (
                        <VoiceTrainPanel />
                    ) : trainingCap.status === "blocked" ? (
                        <TrainingBlockedScreen title={trainingCap.title} reason={trainingCap.reason} />
                    ) : (
                        <div className="flex h-full min-h-0 items-center justify-center p-8 text-sm text-muted-foreground">
                            Checking device capability…
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
