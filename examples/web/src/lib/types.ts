// Shared UI types.

export type Role = "system" | "user" | "model";

export interface ChatMessage {
    role:    Role;
    content: string;
}

export interface SamplingOptions {
    temperature:        number;
    top_k:              number;
    top_p:              number;
    repetition_penalty: number;
    seed:               number;
}

export const DEFAULT_SAMPLING: SamplingOptions = {
    temperature:        0.7,
    top_k:              40,
    top_p:              0.95,
    repetition_penalty: 1.1,
    seed:               0,
};
