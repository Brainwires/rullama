//! fs watcher: notify → debounce → spawn `wasm-pack build` → broadcast
//! `DevEvent::WasmRebuilt` (or `WasmFailed`) to connected WS clients.
//!
//! Debouncing is critical because editor-saves often trigger 2-5 fs events
//! in quick succession; without it we'd kick wasm-pack multiple times in
//! parallel and the WS clients would see contradictory states.

use std::process::Stdio;
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc};
use tokio::time::sleep;

use crate::state::{DevEvent, Paths};

/// Handle returned to `main`. Drop kills the watcher background task.
pub struct WatcherHandle {
    _watcher: RecommendedWatcher,
    _task: tokio::task::JoinHandle<()>,
}

pub fn spawn(paths: Paths, events: broadcast::Sender<DevEvent>) -> WatcherHandle {
    let (notify_tx, notify_rx) = std::sync::mpsc::channel::<notify::Result<Event>>();
    let mut watcher =
        RecommendedWatcher::new(notify_tx, notify::Config::default()).expect("create fs watcher");
    for dir in paths.rust_watch_dirs() {
        if dir.exists() {
            tracing::info!("watching {}", dir.display());
            if let Err(e) = watcher.watch(&dir, RecursiveMode::Recursive) {
                tracing::warn!("watcher.watch({}) failed: {e}", dir.display());
            }
        }
    }

    // Pump the std-channel from notify into a tokio mpsc so the async task
    // can await cleanly. notify's RecommendedWatcher uses a std::sync::mpsc
    // under the hood and we don't want to block a tokio thread on it.
    let (debounce_tx, mut debounce_rx) = mpsc::channel::<()>(64);
    std::thread::spawn(move || {
        for ev in notify_rx.iter() {
            let ev = match ev {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !is_relevant(&ev) {
                continue;
            }
            let _ = debounce_tx.blocking_send(());
        }
    });

    let task = tokio::spawn(async move {
        loop {
            // Block until first event.
            if debounce_rx.recv().await.is_none() {
                break;
            }
            // Drain any further events that fired in the next 300 ms.
            sleep(Duration::from_millis(300)).await;
            while debounce_rx.try_recv().is_ok() {}

            let _ = events.send(DevEvent::WasmBuilding);
            tracing::info!("[watcher] kicking wasm-pack…");
            match build_wasm(&paths).await {
                Ok(_) => {
                    let at_ms = now_ms();
                    tracing::info!("[watcher] wasm-pack OK → broadcasting wasm-rebuilt");
                    let _ = events.send(DevEvent::WasmRebuilt { at_ms });
                }
                Err(tail) => {
                    tracing::warn!(
                        "[watcher] wasm-pack FAILED ({} chars stderr tail)",
                        tail.len()
                    );
                    let _ = events.send(DevEvent::WasmFailed { stderr_tail: tail });
                }
            }
        }
    });

    WatcherHandle {
        _watcher: watcher,
        _task: task,
    }
}

fn is_relevant(ev: &Event) -> bool {
    // Only react to Create/Modify/Remove. Metadata-only access events
    // (e.g. spotlight indexing) are noise.
    matches!(
        ev.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    ) && ev.paths.iter().any(|p| is_rs_or_wgsl(p))
}

fn is_rs_or_wgsl(p: &std::path::Path) -> bool {
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext, "rs" | "wgsl" | "toml")
}

async fn build_wasm(paths: &Paths) -> Result<(), String> {
    // The engine lives in a sibling brainwires checkout now. Build its unified
    // wasm bundle (brainwires-lora = inference Model + training TrainingSession)
    // into the app's pkg/. --out-name rullama keeps the app's /pkg/rullama.js
    // import stable. Without an engine checkout (CI/prod) there's nothing to
    // build — the prebuilt pkg/ from npm/CDN is served as-is.
    let Some(engine) = &paths.engine_dir else {
        return Err("no engine checkout (set BRAINWIRES_ENGINE_DIR or place \
                    ../brainwires-framework/engine); serving prebuilt pkg/ as-is"
            .to_string());
    };
    let mut cmd = Command::new("wasm-pack");
    cmd.current_dir(engine)
        .arg("build")
        .arg("brainwires-lora")
        .arg("--target")
        .arg("web")
        .arg("--release")
        .arg("--out-dir")
        .arg(&paths.pkg_dir)
        .arg("--out-name")
        .arg("rullama")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn().map_err(|e| format!("spawn wasm-pack: {e}"))?;
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| format!("wait: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let combined = {
        let mut s = String::new();
        s.push_str(&String::from_utf8_lossy(&out.stdout));
        s.push_str(&String::from_utf8_lossy(&out.stderr));
        s
    };
    // Tail the last ~2 KB for the browser banner so a giant log doesn't
    // bloat the WS message.
    let tail = if combined.len() > 2048 {
        format!("…(truncated)…\n{}", &combined[combined.len() - 2048..])
    } else {
        combined
    };
    Err(tail)
}

fn now_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
