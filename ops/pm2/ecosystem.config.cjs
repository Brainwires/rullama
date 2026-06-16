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
// PM2 runs `ops/pm2/start.sh`, which REBUILDS web/dist (vite) and then
// `exec`s the release devserver binary — so `pm2 restart` actually picks up
// source edits instead of just relaunching the static server. The server's
// CLI flags (--public, --no-vite, --cors-origins, port) live in that wrapper.
// `setup.sh` builds the binary; this file supervises the wrapper.

const path = require("path");
const repoRoot = path.resolve(__dirname, "..", "..");

module.exports = {
    apps: [
        {
            name: "rullama-devserver",
            script: path.join(repoRoot, "ops/pm2/start.sh"),
            interpreter: "/bin/bash",
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
