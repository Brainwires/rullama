/**
 * rullama cloud proxy — Cloudflare Worker.
 *
 * The production counterpart of the devserver's `/api/cloud/*` route
 * (dev-server/src/cloud.rs). Both Ollama Cloud and OpenAI send
 * NO CORS headers and OpenAI forbids browser-side keys, so a server-side hop
 * is mandatory; this Worker is that hop in Docker/production. nginx
 * reverse-proxies `/api/cloud/*` here (stripping the prefix), so the Worker
 * sees `/{provider}/{chat|models}`.
 *
 * BYOK: the browser sends the user's key in `X-Cloud-Key`; we set
 * `Authorization: Bearer <key>` on the upstream request and stream the
 * (SSE) response straight back. The key is NEVER logged or stored.
 *
 * Billing note: `return new Response(upstream.body, …)` pipes the stream with
 * near-zero CPU — Workers bill CPU time, not the wall-clock spent waiting on
 * the upstream, so a long streamed completion is effectively free.
 *
 * CORS: emitted so the "power-user points their own Worker URL at us" path can
 * call this directly from the browser. The nginx-proxied path is server-side
 * and ignores CORS.
 */

export interface Env {
	/** Comma-separated browser origins allowed to call this Worker directly. */
	CORS_ORIGINS?: string;
}

const UPSTREAM: Record<string, string> = {
	ollama: "https://ollama.com/v1",
	openai: "https://api.openai.com/v1",
};

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const cors = corsHeaders(request.headers.get("Origin"), env);

		if (request.method === "OPTIONS") {
			return new Response(null, { status: 204, headers: cors });
		}

		// nginx strips `/api/cloud/`, so the path is `/{provider}/{action}`.
		const parts = new URL(request.url).pathname.split("/").filter(Boolean);
		const provider = parts[0];
		const action = parts[1];
		const base = provider ? UPSTREAM[provider] : undefined;
		if (!base) return json({ error: `unknown provider: ${provider ?? ""}` }, 400, cors);

		const key = request.headers.get("X-Cloud-Key")?.trim();
		if (!key) return json({ error: "missing X-Cloud-Key" }, 401, cors);

		let upstreamUrl: string;
		let method: string;
		if (action === "chat") {
			upstreamUrl = `${base}/chat/completions`;
			method = "POST";
		} else if (action === "models") {
			upstreamUrl = `${base}/models`;
			method = "GET";
		} else {
			return json({ error: `unknown action: ${action ?? ""}` }, 404, cors);
		}

		// Build a clean upstream request — only the headers we want, the BYOK
		// key as Authorization, the client's JSON body forwarded verbatim.
		const upstreamReq = new Request(upstreamUrl, {
			method,
			headers: {
				authorization: `Bearer ${key}`,
				"content-type": "application/json",
				accept: "text/event-stream",
			},
			body: method === "POST" ? await request.text() : undefined,
		});

		let resp: Response;
		try {
			resp = await fetch(upstreamReq);
		} catch {
			// Do not leak the key or request in the error.
			return json({ error: "cloud upstream unreachable" }, 502, cors);
		}

		// Stream the response through, relaying status + Content-Type and
		// adding CORS + anti-buffering. Drop encoding/length that don't
		// survive re-emission.
		const headers = new Headers(resp.headers);
		for (const [k, v] of Object.entries(cors)) headers.set(k, v);
		headers.set("Cache-Control", "no-cache");
		headers.set("X-Accel-Buffering", "no");
		headers.delete("content-encoding");
		headers.delete("content-length");
		return new Response(resp.body, { status: resp.status, headers });
	},
};

/** CORS headers; `Access-Control-Allow-Origin` is echoed only for an allowed
 *  origin (or `*` in the allowlist). Server-side (nginx) callers send no
 *  Origin and ignore these. */
function corsHeaders(origin: string | null, env: Env): Record<string, string> {
	const allow = (env.CORS_ORIGINS ?? "")
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
	const h: Record<string, string> = {
		"Access-Control-Allow-Methods": "POST, GET, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type, X-Cloud-Key",
		"Access-Control-Max-Age": "86400",
		Vary: "Origin",
	};
	if (origin && (allow.includes(origin) || allow.includes("*"))) {
		h["Access-Control-Allow-Origin"] = origin;
	}
	return h;
}

function json(obj: unknown, status: number, cors: Record<string, string>): Response {
	return new Response(JSON.stringify(obj), {
		status,
		headers: { "content-type": "application/json", ...cors },
	});
}
