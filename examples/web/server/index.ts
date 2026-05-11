// rullama-web API server — Hono on Node.
//
// Phase 1: minimal skeleton that responds to /api/health. Phase 2 will port
// the four endpoints from `examples/pwa/serve.sh` (models / blob / log /
// bench-result) so the React app and the existing static PWA both work
// against this server. Bun is the eventual target runtime; Hono runs on
// both with the same app code.

import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";

const app = new Hono();

app.use("/api/*", cors());

app.get("/api/health", (c) =>
    c.json({ ok: true, ts: Date.now(), runtime: "node+hono" }),
);

// Catch-all so the React dev server's `/api/*` proxy returns a clear 404
// instead of HTML when an endpoint is missing.
app.notFound((c) => c.json({ error: "not found", path: c.req.path }, 404));

const port = Number(process.env.PORT ?? 8088);
serve({ fetch: app.fetch, port }, (info) => {
    console.log(`[rullama-web] api listening on http://0.0.0.0:${info.port}`);
});
