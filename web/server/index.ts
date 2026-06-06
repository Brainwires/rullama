// rullama-web API server — Hono on Node.
//
// Mirrors the four endpoints from the legacy `serve.sh` Python
// dev server, so the React app and the static PWA can both run against this
// process. Bun is the eventual target runtime; Hono runs unchanged on both.
//
//   GET  /api/health         — liveness
//   GET  /api/models         — discover ~/.ollama/models manifests
//   GET  /api/blob/:name     — stream a GGUF blob, supports `Range: bytes=N-M`
//   POST /api/log            — append `{tag, msg, ts}` to /tmp/rullama-page.log
//   POST /api/bench-result   — append a JSON record to /tmp/rullama-bench.jsonl

import fs from "node:fs";
import path from "node:path";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";

import { discoverModels, findBlob, huggingfaceModels } from "./ollama.js";

const PAGE_LOG  = process.env.PAGE_LOG  || "/tmp/rullama-page.log";
const BENCH_LOG = process.env.BENCH_LOG || "/tmp/rullama-bench.jsonl";

const app = new Hono();
app.use("/api/*", cors());

// ───── /api/health ─────────────────────────────────────────────────────
app.get("/api/health", (c) =>
    c.json({ ok: true, ts: Date.now(), runtime: "node+hono" }),
);

// ───── /api/models ─────────────────────────────────────────────────────
// Local Ollama models first; fall back to the public HF list when the
// local scan finds nothing OR `RULLAMA_REMOTE_ONLY=1` forces it. This is
// how the public demo offloads model bandwidth to HF — `rullama.com`
// sets the env var and serves only the HF entries, while devs running
// locally with `~/.ollama/models` keep their multimodal blobs.
app.get("/api/models", (c) => {
    const remoteOnly = process.env.RULLAMA_REMOTE_ONLY === "1";
    const local = remoteOnly ? [] : discoverModels();
    if (local.length > 0) return c.json(local);
    return c.json(huggingfaceModels());
});

// ───── /api/blob/:name ─────────────────────────────────────────────────
// `:name` is "family:tag", URL-encoded by the caller.
app.get("/api/blob/:name", async (c) => {
    const nameTag = decodeURIComponent(c.req.param("name"));
    const blob = findBlob(nameTag);
    if (!blob) {
        return c.json({ error: `model not found: ${nameTag}` }, 404);
    }
    let stat: fs.Stats;
    try { stat = fs.statSync(blob); }
    catch { return c.json({ error: "blob disappeared from disk" }, 500); }
    const size = stat.size;

    // Range support (resumable downloads + fetch ReadableStream).
    let start = 0;
    let end   = size - 1;
    let status = 200;
    const rng = c.req.header("Range") || "";
    if (rng.startsWith("bytes=")) {
        const spec = rng.slice("bytes=".length);
        const m = /^(\d*)-(\d*)$/.exec(spec);
        if (m) {
            if (m[1]) start = Math.max(0, Number(m[1]));
            if (m[2]) end   = Math.min(size - 1, Number(m[2]));
            if (start <= end && end < size) status = 206;
        }
    }
    const length = end - start + 1;

    // Stream the file in 1 MiB chunks. fs.createReadStream supports
    // start/end inclusive byte offsets.
    const fileStream = fs.createReadStream(blob, { start, end, highWaterMark: 1 << 20 });
    const webStream  = new ReadableStream<Uint8Array>({
        start(controller) {
            fileStream.on("data", (chunk: Buffer | string) => {
                if (typeof chunk === "string") chunk = Buffer.from(chunk);
                controller.enqueue(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
            });
            fileStream.on("end", () => controller.close());
            fileStream.on("error", (e) => controller.error(e));
        },
        cancel() { fileStream.destroy(); },
    });

    const headers: Record<string, string> = {
        "Content-Type":   "application/octet-stream",
        "Content-Length": String(length),
        "Accept-Ranges":  "bytes",
        "X-Model-Name":   nameTag,
        "X-Total-Size":   String(size),
    };
    if (status === 206) headers["Content-Range"] = `bytes ${start}-${end}/${size}`;
    return new Response(webStream, { status, headers });
});

// ───── /api/log ────────────────────────────────────────────────────────
// Beacon receiver. `{tag, msg, ts}` JSONL → /tmp/rullama-page.log.
app.post("/api/log", async (c) => {
    let payload: unknown;
    try { payload = await c.req.json(); }
    catch (e) { return c.json({ error: `bad JSON: ${(e as Error).message}` }, 400); }
    fs.appendFileSync(PAGE_LOG, JSON.stringify(payload) + "\n");
    return c.json({});
});

// ───── /api/bench-result ───────────────────────────────────────────────
app.post("/api/bench-result", async (c) => {
    let payload: unknown;
    try { payload = await c.req.json(); }
    catch (e) { return c.json({ error: `bad JSON: ${(e as Error).message}` }, 400); }
    fs.appendFileSync(BENCH_LOG, JSON.stringify(payload) + "\n");
    return c.json({});
});

// In prod, the built React app at `web/dist/` is also served from
// this process. In dev, Vite serves the SPA and proxies /api/* to us.
const distDir = path.resolve(import.meta.dirname ?? __dirname, "..", "dist");
if (fs.existsSync(distDir) && fs.statSync(distDir).isDirectory()) {
    app.use("/*", async (c, next) => {
        const reqPath = c.req.path;
        if (reqPath.startsWith("/api/")) return next();
        // Try the file as-is, else fall back to index.html (SPA routing).
        const tryPath = reqPath === "/" ? "/index.html" : reqPath;
        const filePath = path.join(distDir, tryPath);
        try {
            const st = fs.statSync(filePath);
            if (st.isFile()) {
                const data = fs.readFileSync(filePath);
                const ct = mimeFor(filePath);
                return new Response(data, { headers: { "Content-Type": ct } });
            }
        } catch { /* fall through */ }
        const indexPath = path.join(distDir, "index.html");
        try {
            const data = fs.readFileSync(indexPath);
            return new Response(data, { headers: { "Content-Type": "text/html; charset=utf-8" } });
        } catch {
            return c.text("not found", 404);
        }
    });
}

function mimeFor(p: string): string {
    const ext = path.extname(p).toLowerCase();
    switch (ext) {
        case ".html": return "text/html; charset=utf-8";
        case ".js":   return "text/javascript; charset=utf-8";
        case ".mjs":  return "text/javascript; charset=utf-8";
        case ".css":  return "text/css; charset=utf-8";
        case ".json": return "application/json; charset=utf-8";
        case ".wasm": return "application/wasm";
        case ".svg":  return "image/svg+xml";
        case ".png":  return "image/png";
        case ".jpg":
        case ".jpeg": return "image/jpeg";
        case ".webp": return "image/webp";
        case ".webmanifest": return "application/manifest+json";
        default:      return "application/octet-stream";
    }
}

app.notFound((c) =>
    c.req.path.startsWith("/api/")
        ? c.json({ error: "not found", path: c.req.path }, 404)
        : c.text("not found", 404),
);

const port = Number(process.env.PORT ?? 8088);
serve({ fetch: app.fetch, port }, (info) => {
    console.log(`[rullama-web] api listening on http://0.0.0.0:${info.port}`);
});
