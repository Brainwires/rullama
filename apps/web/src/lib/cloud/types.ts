// Shared types for the opt-in cloud chat backend (Ollama Cloud + OpenAI).

export type CloudProvider = "ollama" | "openai";

/** A loaded cloud chat backend — there is no GPU/worker handle, just the
 *  provider + the upstream model id the proxy forwards to. */
export interface CloudModel {
    provider: CloudProvider;
    /** Upstream model id, e.g. "gemma4:31b" (Ollama) or "gpt-5" (OpenAI). */
    model: string;
}

/** Human label for a provider, for badges/tooltips. */
export function providerLabel(p: CloudProvider): string {
    return p === "ollama" ? "Ollama Cloud" : "OpenAI";
}
