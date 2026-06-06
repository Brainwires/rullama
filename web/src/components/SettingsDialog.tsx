import { Button } from "@/components/ui/button";
import { LogsTab } from "@/components/LogsTab";
import { cn } from "@/lib/utils";
import { usePersistedState } from "@/lib/persisted";
import { HIGH_VRAM_OVERRIDE_KEY } from "@/lib/capability";
import { hardResetAndReload } from "@/lib/restart";
import { RefreshCw } from "lucide-react";

// Hard bounds — also used by App.tsx to normalize old persisted values on
// boot, and by ChatSettings for the per-tab sliders. Keep these conservative;
// users with a real need can edit the JSON in localStorage directly.
// Fallbacks mirror DEFAULT_SAMPLING (lib/types.ts) which tracks Ollama's
// Gemma 4 params{}: temperature 1, top_k 64, top_p 0.95.
export const SETTINGS_BOUNDS = {
    temperature:        { min: 0,    max: 2,    step: 0.05, fallback: 1    },
    top_k:              { min: 0,    max: 200,  step: 1,    fallback: 64   },
    top_p:              { min: 0,    max: 1,    step: 0.01, fallback: 0.95 },
    repetition_penalty: { min: 0.5,  max: 2.0,  step: 0.05, fallback: 1.3  },
    maxTokens:          { min: 16,   max: 4096, step: 16,   fallback: 1024 },
} as const;

type TabKey = "general" | "logs";

/**
 * **Global** settings: the things that aren't specific to a single tab — the High-VRAM override,
 * diagnostic logs, and app-data recovery. Model management moved INTO the tab sidebars (Gemma →
 * Chat, StyleTTS2 → Voice); per-tab generation settings (sampling / system prompt / thinking) live
 * in the Chat right sidebar. Switching to this view neither loads nor unloads any model.
 */
export function SettingsDialog() {
    // Persisted so the App-level "crashed last session" toast can deep-link into Logs.
    const [tab, setTab] = usePersistedState<TabKey>("rullama:settings:tab", "general");
    const activeTab: TabKey = tab === "logs" ? "logs" : "general"; // legacy "sampling"/"voice" → general
    // Manual Premium-tier override — the browser can't read VRAM, so a user with a 24GB+ GPU opts in
    // here to unlock co-resident engines (and any high-VRAM behaviors). See lib/capability.ts.
    const [highVram, setHighVram] = usePersistedState<boolean>(HIGH_VRAM_OVERRIDE_KEY, false);

    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex items-center gap-2 border-b border-border px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Settings
                </span>
            </header>

            <nav role="tablist" aria-label="Settings sections" className="flex shrink-0 border-b border-border px-3">
                <TabButton label="General" active={activeTab === "general"} onClick={() => setTab("general")} />
                <TabButton
                    label="Logs"
                    active={activeTab === "logs"}
                    onClick={() => setTab("logs")}
                    title="Diagnostic logs persisted across sessions (survives crashes)"
                />
            </nav>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                {activeTab === "general" && (
                    <>
                        <section className="flex flex-col gap-2">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Performance</span>
                            <label
                                className="flex items-start gap-2 text-xs text-muted-foreground"
                                title="The browser can't read GPU VRAM. Enable this if you have a 24GB+ GPU to keep the inference and TTS engines loaded together (no reload when switching Chat↔Voice)."
                            >
                                <input
                                    type="checkbox"
                                    checked={highVram}
                                    onChange={(e) => setHighVram(e.target.checked)}
                                    className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer"
                                />
                                <span>
                                    High-VRAM GPU (24&nbsp;GB+) — keep inference + TTS resident together
                                    so switching Chat↔Voice doesn't reload. Only enable if your GPU has the
                                    memory; the browser can't detect it.
                                </span>
                            </label>
                        </section>

                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Trouble</span>
                            <Button
                                variant="outline"
                                size="sm"
                                className="h-8 justify-start gap-2 text-xs"
                                onClick={() => {
                                    if (window.confirm(
                                        "Unregister service worker, clear cached assets, and reload. "
                                        + "Cached models and settings are preserved. Continue?",
                                    )) {
                                        void hardResetAndReload();
                                    }
                                }}
                                title="Wipe service-worker cache and reload — recovery for stuck PWA states"
                            >
                                <RefreshCw className="h-3.5 w-3.5" />
                                Reset app data
                            </Button>
                        </section>
                    </>
                )}

                {activeTab === "logs" && <LogsTab />}
            </div>
        </div>
    );
}

interface TabButtonProps {
    label: string;
    active: boolean;
    onClick: () => void;
    disabled?: boolean;
    title?: string;
}

function TabButton({ label, active, onClick, disabled, title }: TabButtonProps) {
    return (
        <button
            type="button"
            role="tab"
            aria-selected={active}
            disabled={disabled}
            title={title}
            onClick={onClick}
            className={cn(
                "-mb-px border-b-2 px-3 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "disabled:pointer-events-none disabled:opacity-50",
                active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
            )}
        >
            {label}
        </button>
    );
}
