// PM2 ecosystem for the rullama devserver in public/tunnel mode.
//
// Setup:
//   ./ops/pm2/setup.sh
// Then (one-time, for boot survival):
//   sudo pm2 startup launchd -u $USER --hp $HOME
//   pm2 save
//
// Day-to-day:
//   pm2 status
//   pm2 logs rullama-devserver
//   pm2 restart rullama-devserver
//
// We run the release binary DIRECTLY (not via `cargo run`) so PM2
// restart latency is ~tens of ms instead of cargo's startup overhead.
// `setup.sh` builds the binary; this file just supervises it.

const path = require("path");
const repoRoot = path.resolve(__dirname, "..", "..");

module.exports = {
    apps: [
        {
            name: "rullama-devserver",
            script: path.join(
                repoRoot,
                "crates/rullama-devserver/target/release/rullama-devserver",
            ),
            args: [
                "--public",
                "--host", "127.0.0.1",
                "--port", "25321",
                // We're serving the prebuilt dist/, no Vite needed.
                "--no-vite",
                // Whether to also auto-rebuild wasm in the background:
                // safer to leave this on so a code edit pushes through
                // immediately. Cost is ~1 wasm-pack run per save burst.
                // Comment out if you want the public origin to be
                // strictly static.
                // "--no-watch",
                // Origins allowed for cross-origin reads (the page is
                // loaded from this exact hostname per the Cloudflare
                // route). Edit to match your tunnel hostname.
                "--cors-origins", "https://rullama.brainwires.net",
            ],
            cwd: repoRoot,
            env: {
                RUST_LOG: "rullama_devserver=info,tower_http=info,warn",
                // Default page-log dir — pm2 already streams stdout, so
                // this is just for the `/api/log` writes. In public
                // mode we disable /api/log entirely, so this file
                // shouldn't grow.
                RULLAMA_PAGE_LOG: "/tmp/rullama-page.log",
            },
            // Restart policy: restart on any non-zero exit, give it a
            // small backoff so a tight crash loop doesn't pin the CPU.
            autorestart: true,
            max_restarts: 20,
            min_uptime: "10s",
            restart_delay: 2000,
            kill_timeout: 5000,
            // PM2 log file paths. `combine_logs: true` keeps stdout +
            // stderr in one stream so `pm2 logs` is easy to read.
            out_file: path.join(repoRoot, "ops/pm2/rullama-devserver.out.log"),
            error_file: path.join(repoRoot, "ops/pm2/rullama-devserver.err.log"),
            merge_logs: true,
            time: true,
        },
    ],
};
