import { useState } from "react";
import { usePersistedState } from "@/lib/persisted";
import { DOCKED_DEFAULT } from "@/lib/app-helpers";

/**
 * All sidebar open/close state plus the portal-host elements, grouped
 * behind one hook. Each sidebar's open flag is persisted per sidebar
 * (defaulting open when docked); the `*El` values are the DOM hosts that
 * the relevant panels portal their content into.
 *
 *   - history          — Chat-tab left sidebar (conversation list).
 *   - chatSettings     — Chat-tab right sidebar (model + system prompt + sampling + thinking).
 *   - voiceTabSettings — Voice-tab right sidebar (voice picker + clone-model block, portaled from VoicePanel).
 *   - voiceClips       — Voice-tab left sidebar (generated-clips list, portaled from VoicePanel).
 *   - fineTuneSettings — Fine-tune overlay right sidebar (hyperparameters, portaled from FineTunePanel).
 */
export function useSidebars() {
    // Only the left (history) sidebar exists on the Chat tab now; Settings
    // is its own tab so it doesn't compete with chat content for screen
    // real-estate on small displays.
    const [historyOpen, setHistoryOpen] = usePersistedState<boolean>("ui.historyOpen", DOCKED_DEFAULT);
    const [chatSettingsOpen, setChatSettingsOpen] = usePersistedState<boolean>("ui.chatSettingsOpen", DOCKED_DEFAULT);
    const [voiceTabSettingsOpen, setVoiceTabSettingsOpen] = usePersistedState<boolean>("ui.voiceTabSettingsOpen", DOCKED_DEFAULT);
    const [voiceTabSettingsEl, setVoiceTabSettingsEl] = useState<HTMLDivElement | null>(null);
    const [voiceClipsOpen, setVoiceClipsOpen] = usePersistedState<boolean>("ui.voiceClipsOpen", DOCKED_DEFAULT);
    const [voiceClipsEl, setVoiceClipsEl] = useState<HTMLDivElement | null>(null);
    const [fineTuneSettingsOpen, setFineTuneSettingsOpen] = usePersistedState<boolean>("ui.fineTuneSettingsOpen", DOCKED_DEFAULT);
    const [fineTuneSettingsEl, setFineTuneSettingsEl] = useState<HTMLDivElement | null>(null);

    return {
        historyOpen, setHistoryOpen,
        chatSettingsOpen, setChatSettingsOpen,
        voiceTabSettingsOpen, setVoiceTabSettingsOpen,
        voiceTabSettingsEl, setVoiceTabSettingsEl,
        voiceClipsOpen, setVoiceClipsOpen,
        voiceClipsEl, setVoiceClipsEl,
        fineTuneSettingsOpen, setFineTuneSettingsOpen,
        fineTuneSettingsEl, setFineTuneSettingsEl,
    };
}
