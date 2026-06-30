//! Workspace task runner — small dispatcher invoked via `cargo` aliases.
//!
//! New tasks: add a match arm below + the corresponding alias in
//! `.cargo/config.toml`. Keep this dependency-free (std only) so `cargo run
//! -p xtask` stays fast on a cold workspace.

use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let task = match std::env::args().nth(1) {
        Some(t) => t,
        None => {
            eprintln!("usage: cargo <task> (e.g. dev, docker:start)");
            return ExitCode::from(2);
        }
    };
    match task.as_str() {
        "docker:build" => compose(&["build"]),
        "docker:start" => compose(&["up", "-d"]),
        "docker:stop" => compose(&["down"]),
        "docker:restart" => {
            let r = compose(&["build", "--no-cache"]);
            if r != ExitCode::SUCCESS {
                return r;
            }
            compose(&["up", "-d", "--force-recreate"])
        }
        "docker:logs" => compose(&["logs", "-f", "--tail=200"]),
        "docker:ps" => compose(&["ps"]),
        "dev" => dev(),
        other => {
            eprintln!("xtask: unknown task `{other}`");
            eprintln!(
                "tasks: dev, docker:build, docker:start, docker:stop, docker:restart, docker:logs, docker:ps"
            );
            ExitCode::from(2)
        }
    }
}

/// `cargo dev` — bring up the native dev server (replaces the Python
/// serve-tunnel.sh / serve-iphone.sh scripts). Forwards any trailing
/// argv to the binary so flags like `--no-vite` work.
fn dev() -> ExitCode {
    // rullama-devserver is excluded from the workspace (see Cargo.toml
    // exclude list — keeps it out of `cargo build --workspace --target
    // wasm32-unknown-unknown`). Run it via --manifest-path.
    let manifest = Path::new("services/dev-server/Cargo.toml");
    if !manifest.is_file() {
        eprintln!(
            "xtask: {} not found; are you in the repo root?",
            manifest.display()
        );
        return ExitCode::from(1);
    }
    let forwarded: Vec<String> = std::env::args().skip(2).collect();
    // In --public mode the devserver STATIC-serves web/dist/ (no Vite HMR), so a
    // stale or half-built dist 404s its files — e.g. a page reloaded mid-build
    // sees the new hashed JS but a not-yet-written sw.js. Build dist fresh first
    // so what's served always matches source. Local-dev mode reverse-proxies to
    // Vite (builds on the fly), so there's nothing to pre-build there.
    if forwarded.iter().any(|a| a == "--public") {
        eprintln!("$ pnpm -C apps/web build   (refreshing web/dist/ for the --public static serve)");
        match Command::new("pnpm").args(["-C", "apps/web", "build"]).status() {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!(
                "xtask: web build failed (exit {}); serving the existing web/dist/ as-is",
                s.code().unwrap_or(-1)
            ),
            Err(e) => eprintln!(
                "xtask: could not run `pnpm -C apps/web build` ({e}); serving the existing web/dist/ as-is"
            ),
        }
    }
    eprintln!(
        "$ cargo run -q --manifest-path {} --release --{}",
        manifest.display(),
        if forwarded.is_empty() {
            "".into()
        } else {
            format!(" {}", forwarded.join(" "))
        }
    );
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("-q")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--release")
        .arg("--");
    if !forwarded.is_empty() {
        cmd.args(&forwarded);
    }
    match cmd.status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("xtask: failed to spawn cargo for rullama-devserver: {e}");
            ExitCode::from(127)
        }
    }
}

fn compose(args: &[&str]) -> ExitCode {
    eprintln!("$ docker compose {}", args.join(" "));
    // Set RULLAMA_COMMIT from `git rev-parse --short HEAD` if not already
    // exported by the caller. compose.yaml interpolates this into the
    // image's build-args so emit-version.mjs can stamp the commit into
    // /version.json + __APP_VERSION__. Failure is fine — emit-version
    // falls back to "nogit" and the timestamp is still unique.
    let mut cmd = Command::new("docker");
    cmd.arg("compose").args(args);
    if std::env::var_os("RULLAMA_COMMIT").is_none()
        && let Ok(output) = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
        && output.status.success()
    {
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !sha.is_empty() {
            cmd.env("RULLAMA_COMMIT", &sha);
            eprintln!("(setting RULLAMA_COMMIT={sha})");
        }
    }
    let status = cmd.status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("xtask: failed to spawn docker: {e}");
            ExitCode::from(127)
        }
    }
}
