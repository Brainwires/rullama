import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { readFileSync } from "node:fs";
import path from "node:path";

// The wasm bundle (`/pkg/rullama.js` + `rullama_bg.wasm`) is built by
// `wasm-pack` into the project root's `pkg/` directory. Vite serves it
// via a static-asset alias so the bundle's `import.meta.url`-based
// wasm fetch resolves to `/pkg/rullama_bg.wasm` in both dev and prod.
const repoRoot = path.resolve(__dirname, "..", "..");

// Read the build version emitted by `scripts/emit-version.mjs`. This
// runs as part of the `build` npm script *before* Vite, so the file
// exists when this config is evaluated. In `dev` mode the file may
// not exist yet — fall back to "dev" so the bundle still type-checks.
// The same version string is served at `/version.json`, and the
// runtime update-check compares the two.
function readBuildVersion(): string {
    try {
        const raw = readFileSync(path.resolve(__dirname, "public", "version.json"), "utf8");
        const j = JSON.parse(raw) as { version?: string };
        return j.version ?? "dev";
    } catch { return "dev"; }
}
const BUILD_VERSION = readBuildVersion();

export default defineConfig({
    root: __dirname,
    plugins: [
        react(),
        VitePWA({
            // We ship our own manifest at public/manifest.webmanifest so iOS
            // Safari "Add to Home Screen" gets the icons we control. Tell the
            // plugin to leave manifest generation alone.
            manifest: false,
            // "prompt" mode: the SW installs in the background but does
            // NOT auto-activate. The user-facing "an update is available"
            // signal is the version manifest (lib/version.ts) + the
            // in-app banner; clicking Apply triggers a coordinated
            // multi-tab shutdown + reload, which is when the new SW
            // takes over. Workbox's `skipWaiting` would race that flow
            // (silently activating a SW whose precache no longer
            // matches the running page's chunk URLs), so it's off.
            registerType: "prompt",
            // Only precache the small static shell — the 7 GB GGUF is *not*
            // precached, it lives in OPFS via our own writer worker.
            workbox: {
                // No skipWaiting: the new SW waits until all clients are
                // gone (which happens during the coordinated reload).
                // clientsClaim stays on so the new SW immediately
                // handles fetches once it activates.
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
                // **cacheId bumped on every build.** Suffixed with the
                // build timestamp so each `vite build` invalidates ALL
                // previously cached SW buckets — no more stale-wasm
                // archaeology after a rebuild. Combined with the
                // NetworkFirst /pkg/ handler below, this makes any
                // cache-related staleness structurally impossible:
                // the bucket name is unique to this build AND the
                // wasm handler prefers network anyway.
                //
                // History: we lost ~6 iPhone test cycles in a row to a
                // stale wasm because the previous static `rullama-v6` +
                // CacheFirst (30-day TTL) silently served bundles built
                // by older commits. Never again.
                cacheId: `rullama-${BUILD_VERSION}-${Date.now()}`,
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
                    // **wasm bundle: NetworkFirst with a generous cache
                    // fallback.** Previously this was CacheFirst with a
                    // 30-day TTL, which meant: a new wasm-pack build was
                    // INVISIBLE to any iPhone (or any user) who'd already
                    // fetched the prior bundle, for up to 30 days. The
                    // page log showed stale beacons and made debugging
                    // training crashes much harder — every iteration
                    // since `5487cb8` was actually running 5487cb8's
                    // wasm until v7 of this cache rolled out. This burned
                    // ~6 commits worth of iPhone test cycles silently.
                    //
                    // NetworkFirst with networkTimeoutSeconds=5 means:
                    //   • Online (always our case): fetch network → cache
                    //     stays warm but is never served if network's up.
                    //     Rebuilds are visible on next page load.
                    //   • Slow network: 5 s timeout falls back to cache,
                    //     so users don't sit on a blank page.
                    //   • Offline (PWA edge case): cache hit, app boots
                    //     against last-known-good bundle.
                    // The bundle is ~4 MB so network re-fetch on each
                    // navigation is cheap (~30-50ms on broadband).
                    {
                        urlPattern: /\/pkg\/.*\.(js|wasm)$/,
                        handler:    "NetworkFirst",
                        options: {
                            cacheName: "rullama-wasm-v2",
                            networkTimeoutSeconds: 5,
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
    define: {
        // Same version string as `public/version.json`. The runtime
        // boot check (`lib/version.ts`) compares the two; a mismatch
        // means a newer build is deployed and the app surfaces the
        // update banner.
        __APP_VERSION__: JSON.stringify(BUILD_VERSION),
    },
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
