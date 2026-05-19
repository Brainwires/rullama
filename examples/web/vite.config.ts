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
            // "autoUpdate" + Workbox `skipWaiting`/`clientsClaim` is the
            // silent-update path. The dialog-driven "prompt" mode forced the
            // user through a reload after the app had already mounted —
            // confusing and slow. Now we gate React mount on SW freshness
            // (see `src/lib/pwa.ts::ensureFreshServiceWorker`), so the new
            // SW is in charge before the user sees anything.
            registerType: "autoUpdate",
            // Only precache the small static shell — the 7 GB GGUF is *not*
            // precached, it lives in OPFS via our own writer worker.
            workbox: {
                // New SW takes over as soon as it's installed. Navigation
                // requests are handled NetworkFirst (below) so even a
                // mid-session SW swap doesn't strand the page on stale
                // chunk URLs the new precache no longer has.
                skipWaiting: true,
                clientsClaim: true,
                // `cleanupOutdatedCaches` evicts cache buckets whose
                // *prefix* no longer matches; bumping `cacheId` is what
                // changes the prefix. Workbox precaches by URL+revision
                // and refuses to refetch when both match — even if the
                // server response (e.g. Content-Type) has since changed.
                // Bump this suffix any time a deploy needs every device
                // to drop the precache and start clean.
                //   docs: https://developer.chrome.com/docs/workbox/modules/workbox-precaching
                //   issue: https://github.com/GoogleChrome/workbox/issues/2757
                cacheId: "rullama-v5",
                cleanupOutdatedCaches: true,
                // index.html is intentionally OMITTED from precache: it
                // goes through the NetworkFirst navigation handler below
                // so every reload picks up the live deploy's HTML. This
                // is the load-bearing fix for the post-deploy black
                // screen — without it, the old SW served stale index.html
                // referencing chunk hashes the new SW had just deleted,
                // and the page wedged trying to load 404'd chunks until
                // the 8 s watchdog kicked in.
                globPatterns: ["**/*.{css,js,svg,png,webmanifest}"],
                globIgnores:  ["**/pkg/**", "**/*.wasm", "**/index.html"],
                runtimeCaching: [
                    // Navigation: NetworkFirst. nginx-rullama.conf already
                    // serves HTML with `no-store, no-cache, must-revalidate,
                    // max-age=0` (via the $rullama_cache_ctrl map), so the
                    // network response is always fresh. Cache fallback
                    // (rullama-pages) is purely for offline reloads — if
                    // the network is unreachable we serve the last good
                    // HTML; if it's just slow we wait up to 3 s.
                    {
                        urlPattern: ({ request }: { request: Request }) =>
                            request.mode === "navigate",
                        handler: "NetworkFirst",
                        options: {
                            cacheName: "rullama-pages",
                            networkTimeoutSeconds: 3,
                            expiration: { maxEntries: 5 },
                            cacheableResponse: { statuses: [0, 200] },
                        },
                    },
                    // The wasm bundle is large; we don't want Workbox to
                    // inline-precache it but we *do* want it cached on
                    // first fetch.
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
                // navigateFallback explicitly null so vite-plugin-pwa does
                // NOT auto-register a NavigationRoute(createHandlerBoundToURL).
                // That auto-route would be registered FIRST and short-
                // circuit our NetworkFirst nav handler above. The
                // NetworkFirst handler already covers SPA deep links —
                // nginx returns index.html for any non-asset path via
                // `try_files $uri $uri/ /index.html;` (nginx-rullama.conf:103),
                // so the React router sees the same HTML regardless of
                // which path the user navigated to.
                navigateFallback: null,
            },
            devOptions: {
                // Off by default; turn on with `VITE_PWA_DEV=1 pnpm dev`. The
                // service worker can otherwise confuse rapid wasm-pack rebuilds.
                enabled: !!process.env.VITE_PWA_DEV,
                type: "module",
            },
            includeAssets: [
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
