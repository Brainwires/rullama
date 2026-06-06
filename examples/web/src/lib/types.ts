// Shared UI types.

export type Role = "system" | "user" | "model";

export interface ImageAttachment {
    /** Channel-first f32 [-1, 1] tensor — what `encodeImage` consumes.
     *  Present only for in-session attachments (the user just picked
     *  the file). Reloaded-from-history images have only the thumbnail. */
    pixels?: Float32Array;
    h:       number;
    w:       number;
    /** Small JPEG thumbnail — used to render the image inline in the
     *  user message bubble + the input-row preview. Persisted to
     *  SQLite alongside the message so reloaded conversations restore. */
    dataUrl: string;
}

export interface ChatMessage {
    role:    Role;
    content: string;
    /** Optional image attachments. Only set in-session on user turns
     *  that came with files. Not persisted to SQLite — a chat reloaded
     *  from history shows the message text alone. */
    images?: ImageAttachment[];
}

export interface SamplingOptions {
    temperature:        number;
    top_k:              number;
    top_p:              number;
    repetition_penalty: number;
    seed:               number;
}

// Defaults mirror Ollama's Gemma 4 params{} (temperature 1, top_k 64,
// top_p 0.95) so out-of-the-box sampling matches the reference engine.
// repetition_penalty stays 1.3 (Ollama sets none; 1.3 is the tuned value
// the garlic-LoRA recipe relies on). Keep in sync with SETTINGS_BOUNDS
// fallbacks in SettingsDialog.tsx.
export const DEFAULT_SAMPLING: SamplingOptions = {
    temperature:        1,
    top_k:              64,
    top_p:              0.95,
    repetition_penalty: 1.3,
    seed:               0,
};

export const DEFAULT_SYSTEM_PROMPT =
    "You are a helpful AI chat assistant.";
