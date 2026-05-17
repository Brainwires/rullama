//! Workspace task runner — small dispatcher invoked via `cargo` aliases.
//!
//! New tasks: add a match arm below + the corresponding alias in
//! `.cargo/config.toml`. Keep this dependency-free (std only) so `cargo run
//! -p xtask` stays fast on a cold workspace.

use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let task = match std::env::args().nth(1) {
        Some(t) => t,
        None => { eprintln!("usage: cargo <task> (e.g. docker:start)"); return ExitCode::from(2); }
    };
    match task.as_str() {
        "docker:build"   => compose(&["build"]),
        "docker:start"   => compose(&["up", "-d"]),
        "docker:stop"    => compose(&["down"]),
        "docker:restart" => {
            let r = compose(&["build", "--no-cache"]);
            if r != ExitCode::SUCCESS { return r; }
            compose(&["up", "-d", "--force-recreate"])
        }
        "docker:logs"    => compose(&["logs", "-f", "--tail=200"]),
        "docker:ps"      => compose(&["ps"]),
        other => {
            eprintln!("xtask: unknown task `{other}`");
            eprintln!("tasks: docker:build, docker:start, docker:stop, docker:restart, docker:logs, docker:ps");
            ExitCode::from(2)
        }
    }
}

fn compose(args: &[&str]) -> ExitCode {
    eprintln!("$ docker compose {}", args.join(" "));
    let status = Command::new("docker").arg("compose").args(args).status();
    match status {
        Ok(s) if s.success()           => ExitCode::SUCCESS,
        Ok(s)                          => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e)                         => { eprintln!("xtask: failed to spawn docker: {e}"); ExitCode::from(127) }
    }
}
