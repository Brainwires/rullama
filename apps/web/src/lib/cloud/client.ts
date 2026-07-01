// Browser cloud chat client — streams OpenAI-compatible chat completions
// THROUGH the same-origin proxy (`/api/cloud/{provider}/chat`), never directly
// to the provider (both block browser CORS). Pure main-thread fetch + SSE
// parsing; no GPU, no worker, no KV cache.

import type { SamplingOptions } from "@/lib/types";
import type { CloudProvider } from "./types";

export interface CloudChatMessage {
    role: "system" | "user" | "assistant";
    content: string;
}

export interface CloudChatArgs {
    provider: CloudProvider;
    model: string;
    apiKey: string;
    messages: CloudChatMessage[];
    sampling: SamplingOptions;
    maxTokens: number;
    /** Power-user override: a base URL pointing at the user's OWN Cloudflare
     *  Worker (which emits CORS and routes on `/{provider}/chat`). When unset,
     *  we call the same-origin `/api/cloud/{provider}/chat` path served by the
     *  devserver (dev) or nginx→Worker (prod). */
    proxyBase?: string;
    signal?: AbortSignal;
}

/** A failed cloud request — carries the upstream status + best-effort message
 *  (provider error body) for a UI toast. */
export class CloudError extends Error {
    constructor(
        public status: number,
        message: string,
    ) {
        super(message);
        this.name = "CloudError";
    }
}

/**
 * Build the OpenAI-compatible request body, mapping rullama's SamplingOptions
 * to the subset each provider safely accepts.
 *
 * Documented v1 fidelity gaps (vs the local engine):
 *  - **Ollama** gets full fidelity: temperature, top_p, top_k, seed, max_tokens
 *    (its OpenAI-compat endpoint accepts them).
 *  - **OpenAI** gets a CONSERVATIVE set: only stream, max_completion_tokens, and
 *    seed. The gpt-5.x reasoning models reject non-default temperature/top_p
 *    with a 400, so we omit them (the provider's defaults apply) rather than
 *    risk a hard failure. top_k has no OpenAI equivalent.
 *  - **repetition_penalty** is dropped for BOTH — OpenAI uses frequency/presence
 *    penalties on a different scale; a faithful remap is a follow-up.
 */
function buildBody(a: CloudChatArgs): Record<string, unknown> {
    const body: Record<string, unknown> = {
        model: a.model,
        messages: a.messages,
        stream: true,
    };
    if (a.sampling.seed && a.sampling.seed !== 0) body.seed = a.sampling.seed;
    if (a.provider === "ollama") {
        body.temperature = a.sampling.temperature;
        body.top_p = a.sampling.top_p;
        body.top_k = a.sampling.top_k;
        if (a.maxTokens > 0) body.max_tokens = a.maxTokens;
    } else {
        // OpenAI gpt-5.x: max_completion_tokens (not max_tokens); leave
        // temperature/top_p to the model's defaults to avoid reasoning-model 400s.
        if (a.maxTokens > 0) body.max_completion_tokens = a.maxTokens;
    }
    return body;
}

function chatUrl(a: CloudChatArgs): string {
    // Override → the user's own Worker, which routes on `/{provider}/chat`.
    // Default → same-origin proxy path (`/api/cloud/{provider}/chat`).
    if (a.proxyBase && a.proxyBase.trim()) {
        return `${a.proxyBase.trim().replace(/\/+$/, "")}/${a.provider}/chat`;
    }
    return `/api/cloud/${a.provider}/chat`;
}

/**
 * Stream a cloud chat completion, yielding text deltas as they arrive. Throws
 * `CloudError` on a non-2xx response (with the provider's error message).
 */
export async function* streamCloudChat(a: CloudChatArgs): AsyncGenerator<string> {
    const resp = await fetch(chatUrl(a), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-Cloud-Key": a.apiKey,
        },
        body: JSON.stringify(buildBody(a)),
        signal: a.signal,
    });

    if (!resp.ok || !resp.body) {
        throw new CloudError(resp.status, await errorMessage(resp));
    }
    // Guard against a 200 that ISN'T the proxy — e.g. on `cargo dev` the request
    // falling through to the Vite SPA fallback returns index.html (200, HTML).
    // Without this the SSE reader would block forever with no error. The proxy
    // always relays the provider's text/event-stream (or application/json error).
    const ct = (resp.headers.get("content-type") || "").toLowerCase();
    if (ct.includes("text/html")) {
        throw new CloudError(
            resp.status,
            "Cloud proxy not reachable (got an HTML page). If you're running `cargo dev`, restart it so the /api/cloud/* route is served.",
        );
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE: line-delimited `data: {json}` frames; `data: [DONE]` ends it.
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (data === "[DONE]") return;
            try {
                const delta = JSON.parse(data)?.choices?.[0]?.delta?.content;
                if (typeof delta === "string" && delta.length > 0) yield delta;
            } catch {
                // keepalive comment or a split frame — ignore.
            }
        }
    }
}

/** Best-effort extraction of a human message from a provider error body. */
async function errorMessage(resp: Response): Promise<string> {
    try {
        const txt = await resp.text();
        try {
            const j = JSON.parse(txt);
            return j?.error?.message ?? j?.error ?? txt ?? `HTTP ${resp.status}`;
        } catch {
            return txt || `HTTP ${resp.status}`;
        }
    } catch {
        return `HTTP ${resp.status}`;
    }
}
