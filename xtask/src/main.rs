//! Workspace task runner — small dispatcher invoked via `cargo` aliases.
//!
//! New tasks: add a match arm below + the corresponding alias in
//! `.cargo/config.toml`. Keep this dependency-free (std only) so `cargo run
//! -p xtask` stays fast on a cold workspace.

use std::fs;
use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let task = match std::env::args().nth(1) {
        Some(t) => t,
        None => {
            eprintln!("usage: cargo <task> (e.g. docker:start, bump 0.3.0)");
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
        "bump" => bump(),
        other => {
            eprintln!("xtask: unknown task `{other}`");
            eprintln!(
                "tasks: docker:build, docker:start, docker:stop, docker:restart, docker:logs, docker:ps, bump"
            );
            ExitCode::from(2)
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
    if std::env::var_os("RULLAMA_COMMIT").is_none() {
        if let Ok(output) = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
        {
            if output.status.success() {
                let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !sha.is_empty() {
                    cmd.env("RULLAMA_COMMIT", &sha);
                    eprintln!("(setting RULLAMA_COMMIT={sha})");
                }
            }
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

/// `cargo bump <new-version>` — edit `version = ...` in both crate
/// manifests, plus the `rullama = { path, version = "MAJOR.MINOR" }`
/// constraint in `rullama-finetune` (cargo caret-resolves that against
/// crates.io after `rullama` is published).
fn bump() -> ExitCode {
    let new_version = match std::env::args().nth(2) {
        Some(v) => v,
        None => {
            eprintln!("usage: cargo bump <new-version>  (e.g. 0.3.0)");
            return ExitCode::from(2);
        }
    };
    let (nmaj, nmin, _npat) = match parse_semver(&new_version) {
        Some(v) => v,
        None => {
            eprintln!("bump: `{new_version}` is not a valid MAJOR.MINOR.PATCH semver");
            return ExitCode::from(2);
        }
    };
    let new_mm = format!("{nmaj}.{nmin}");

    let rullama_path = Path::new("crates/rullama/Cargo.toml");
    let finetune_path = Path::new("crates/rullama-finetune/Cargo.toml");

    let old_version = match read_version(rullama_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bump: {e}");
            return ExitCode::from(1);
        }
    };
    if old_version == new_version {
        eprintln!("bump: version already at {new_version}, nothing to do");
        return ExitCode::SUCCESS;
    }
    let (omaj, omin, _opat) = match parse_semver(&old_version) {
        Some(v) => v,
        None => {
            eprintln!("bump: current rullama version `{old_version}` is not parseable semver");
            return ExitCode::from(1);
        }
    };
    let old_mm = format!("{omaj}.{omin}");

    eprintln!("bumping {old_version} → {new_version}");

    // 1) crates/rullama/Cargo.toml — one `version = "<X>"` near the top.
    if let Err(e) = replace_in_file(
        rullama_path,
        &format!("version          = \"{old_version}\""),
        &format!("version          = \"{new_version}\""),
        1,
    ) {
        eprintln!("bump: {e}");
        return ExitCode::from(1);
    }

    // 2) crates/rullama-finetune/Cargo.toml — its own `version = "<X>"` AND
    //    the path-dep constraint on rullama (MAJOR.MINOR only).
    if let Err(e) = replace_in_file(
        finetune_path,
        &format!("version     = \"{old_version}\""),
        &format!("version     = \"{new_version}\""),
        1,
    ) {
        eprintln!("bump: {e}");
        return ExitCode::from(1);
    }

    if old_mm != new_mm {
        if let Err(e) = replace_in_file(
            finetune_path,
            &format!("rullama = {{ path = \"../rullama\", version = \"{old_mm}\" }}"),
            &format!("rullama = {{ path = \"../rullama\", version = \"{new_mm}\" }}"),
            1,
        ) {
            eprintln!("bump: {e}");
            return ExitCode::from(1);
        }
        eprintln!("  ↳ rullama-finetune path-dep constraint: {old_mm} → {new_mm}");
    }

    eprintln!("bump: done. next: review the diff, commit, then `./scripts/publish.sh`.");
    ExitCode::SUCCESS
}

/// Parse "MAJOR.MINOR.PATCH" into integers. No prerelease / build metadata.
fn parse_semver(s: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// Read the first top-level `version = "..."` line from a Cargo.toml.
fn read_version(path: &Path) -> Result<String, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    for line in content.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("version") {
            let r = rest.trim_start();
            if let Some(after_eq) = r.strip_prefix("=") {
                let v = after_eq.trim().trim_matches('"');
                // Skip `version.workspace = true` and similar.
                if !v.contains('.') && v != "true" {
                    continue;
                }
                if v == "true" {
                    continue;
                }
                return Ok(v.to_string());
            }
        }
    }
    Err(format!(
        "no `version = \"...\"` line found in {}",
        path.display()
    ))
}

/// Exact string replace, with an `expected` match-count guard so we don't
/// silently no-op or over-replace if the manifest layout changes.
fn replace_in_file(path: &Path, from: &str, to: &str, expected: usize) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let actual = content.matches(from).count();
    if actual != expected {
        return Err(format!(
            "{}: expected {expected} match(es) for `{from}`, found {actual} — manifest layout may have shifted",
            path.display(),
        ));
    }
    let new_content = content.replace(from, to);
    fs::write(path, new_content).map_err(|e| format!("write {}: {e}", path.display()))
}
