import { Button } from "@/components/ui/button";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { LogsTab } from "@/components/LogsTab";
import { type ModelEntry } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePersistedState } from "@/lib/persisted";
import { hardResetAndReload } from "@/lib/restart";
import { RefreshCw } from "lucide-react";

// Hard bounds — also used by App.tsx to normalize old persisted values on
// boot, and by ChatSettings for the per-tab sliders. Keep these conservative;
// users with a real need can edit the JSON in localStorage directly.
export const SETTINGS_BOUNDS = {
    temperature:        { min: 0,    max: 2,    step: 0.05, fallback: 0.7  },
    top_k:              { min: 0,    max: 200,  step: 1,    fallback: 40   },
    top_p:              { min: 0,    max: 1,    step: 0.01, fallback: 0.95 },
    repetition_penalty: { min: 0.5,  max: 2.0,  step: 0.05, fallback: 1.1  },
    maxTokens:          { min: 16,   max: 4096, step: 16,   fallback: 1024 },
} as const;

interface Props {
    modelStatus:    ModelStatus;
    loadingPercent: number;
    loadingLabel:   string;
    statusText:     string;
    onLoadModel:    (m: ModelEntry) => void;
    onDeleteModel:  (m: ModelEntry) => void;
    onEjectModel:   () => void;
}

type TabKey = "general" | "logs";

/**
 * **Global** settings: model management, diagnostic logs, and app-data recovery — the things
 * that aren't specific to a single tab. Per-tab generation settings (sampling / system prompt /
 * thinking) live in the Chat right sidebar; voice settings in the Voice/Fine-tune sidebars.
 */
export function SettingsDialog(props: Props) {
    // Persisted so the App-level "crashed last session" toast can deep-link into Logs.
    const [tab, setTab] = usePersistedState<TabKey>("rullama:settings:tab", "general");
    const activeTab: TabKey = tab === "logs" ? "logs" : "general"; // legacy "sampling"/"voice" → general

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
                        <section className="flex flex-col gap-1.5">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Model</span>
                            <ModelLoader
                                status={props.modelStatus}
                                loadingPercent={props.loadingPercent}
                                loadingLabel={props.loadingLabel}
                                statusText={props.statusText}
                                onLoad={props.onLoadModel}
                                onDelete={props.onDeleteModel}
                                onEject={props.onEjectModel}
                            />
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
