import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import path from "node:path";

// The wasm bundle (`/pkg/rullama.js` + `rullama_bg.wasm`) is built by
// `wasm-pack` into the project root's `pkg/` directory. Vite serves it
// via a static-asset alias so the bundle's `import.meta.url`-based
// wasm fetch resolves to `/pkg/rullama_bg.wasm` in both dev and prod.
const repoRoot = path.resolve(__dirname, "..", "..");

export default defineConfig({
    root: __dirname,
    plugins: [
        react(),
        VitePWA({
            // We ship our own manifest at public/manifest.webmanifest so iOS
            // Safari "Add to Home Screen" gets the icons we control. Tell the
            // plugin to leave manifest generation alone.
            manifest: false,
            registerType: "autoUpdate",
            // Only precache the small static shell — the 7 GB GGUF is *not*
            // precached, it lives in OPFS via our own writer worker.
            workbox: {
                globPatterns: ["**/*.{html,css,js,svg,png,webmanifest}"],
                globIgnores:  ["**/pkg/**", "**/*.wasm"],
                // The wasm bundle is large; we don't want Workbox to inline-
                // precache it but we *do* want it cached on first fetch.
                runtimeCaching: [
                    {
                        urlPattern: /\/pkg\/.*\.(js|wasm)$/,
                        handler:    "CacheFirst",
                        options: {
                            cacheName: "rullama-wasm",
                            expiration: { maxEntries: 4, maxAgeSeconds: 60 * 60 * 24 * 30 },
                            cacheableResponse: { statuses: [0, 200] },
                        },
                    },
                ],
                // Allow service-worker control over a 30 MB precache budget
                // (the placeholder is ~250 kB; real icons later).
                maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
                // SPA fallback so deep links to "/anything" still load the app.
                navigateFallback: "/index.html",
                navigateFallbackDenylist: [/^\/api\//],
            },
            devOptions: {
                // Off by default; turn on with `VITE_PWA_DEV=1 pnpm dev`. The
                // service worker can otherwise confuse rapid wasm-pack rebuilds.
                enabled: !!process.env.VITE_PWA_DEV,
                type: "module",
            },
            includeAssets: [
                "favicon.svg",
                "icons/icon-192.png",
                "icons/icon-512.png",
                "icons/icon-mask-512.png",
                "manifest.webmanifest",
            ],
        }),
    ],
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
