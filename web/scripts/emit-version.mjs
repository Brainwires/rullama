// emit-version.mjs — write public/version.json with build timestamp + git short-hash.
//
// Runs before `vite build`. Writes `public/version.json` with the
// build's identifying triple `{ version, builtAt, commit }`. vite.config.ts
// reads the same file at config-time and uses the `version` string in
// its `define: { __APP_VERSION__ }` block, so the bundle and the
// served `/version.json` are paired.
//
// Inside Docker we don't have a git checkout (stage 2 only has source
// files copied in), so `git rev-parse` falls back to env var
// RULLAMA_COMMIT, then "nogit" — the timestamp half still uniquely
// identifies the build.

import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const projectDir = resolve(__dirname, "..");
const publicDir  = resolve(projectDir, "public");
const outFile    = resolve(publicDir, "version.json");

function gitShort() {
    // Pull the short SHA from RULLAMA_COMMIT (set by CI / docker build
    // arg) first; fall back to `git rev-parse` if a checkout is around.
    if (process.env.RULLAMA_COMMIT) {
        return process.env.RULLAMA_COMMIT.replace(/[^A-Za-z0-9-]/g, "").slice(0, 12) || "nogit";
    }
    try {
        return execSync("git rev-parse --short HEAD", {
            cwd: projectDir,
            stdio: ["ignore", "pipe", "ignore"],
        }).toString().trim() || "nogit";
    } catch { return "nogit"; }
}

const now = new Date();
const pad = (n) => String(n).padStart(2, "0");
const stamp =
    String(now.getUTCFullYear()) +
    pad(now.getUTCMonth() + 1) +
    pad(now.getUTCDate()) +
    "-" +
    pad(now.getUTCHours()) +
    pad(now.getUTCMinutes()) +
    pad(now.getUTCSeconds());

const commit  = gitShort();
const version = `${stamp}-${commit}`;
const payload = {
    version,
    builtAt: now.toISOString(),
    commit,
};

mkdirSync(publicDir, { recursive: true });
writeFileSync(outFile, JSON.stringify(payload, null, 2) + "\n", "utf8");

console.error(`[emit-version] wrote ${outFile} (version=${version})`);
