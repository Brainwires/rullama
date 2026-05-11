import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The wasm bundle (`/pkg/rullama.js` + `rullama_bg.wasm`) is built by
// `wasm-pack` into the project root's `pkg/` directory. Vite serves it
// via a static-asset alias so the bundle's `import.meta.url`-based
// wasm fetch resolves to `/pkg/rullama_bg.wasm` in both dev and prod.
const repoRoot = path.resolve(__dirname, "..", "..");

export default defineConfig({
    root: __dirname,
    plugins: [react()],
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "src"),
            // Re-expose the wasm-pack output under a stable URL.
            "/pkg": path.resolve(repoRoot, "pkg"),
        },
    },
    server: {
        port: 5173,
        host: true,
        fs: {
            // Allow serving files from project root (so /pkg/ works).
            allow: [repoRoot],
        },
        proxy: {
            "/api": {
                target: "http://localhost:8088",
                changeOrigin: true,
            },
        },
    },
    // Vite handles workers when imported via `?worker` or `new URL(...)`.
    worker: { format: "es" },
    build: {
        outDir: "dist",
        target: "es2022",
        sourcemap: true,
    },
});
