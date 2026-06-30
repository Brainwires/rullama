//! Shared types: app state passed to every axum handler + the dev-event
//! broadcast payload exchanged between the watcher and WS subscribers.

use std::path::{Path, PathBuf};

use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Clone, Debug)]
pub struct Paths {
    pub repo_root: PathBuf,
    pub pkg_dir: PathBuf,
    pub ollama_models: PathBuf,
    /// The rullama-framework engine sub-workspace (sibling checkout). The inference
    /// engine lives in a separate repo now; for local dev we build its wasm
    /// bundle from here into `pkg_dir`. `None` when no engine checkout is
    /// present — then the devserver serves a prebuilt `pkg/` (CI/prod, sourced
    /// from npm/CDN) and the watcher does not rebuild.
    pub engine_dir: Option<PathBuf>,
}

impl Paths {
    pub fn resolve(
        repo_root: Option<PathBuf>,
        ollama_models: Option<PathBuf>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let repo_root = match repo_root {
            Some(p) => p,
            None => find_repo_root()?,
        };
        let pkg_dir = repo_root.join("pkg");
        let ollama_models = ollama_models
            .or_else(|| std::env::var_os("OLLAMA_MODELS").map(PathBuf::from))
            .unwrap_or_else(|| {
                dirs_home()
                    .map(|h| h.join(".ollama").join("models"))
                    .unwrap_or_else(|| PathBuf::from("/Users/nightness/.ollama/models"))
            });
        // RULLAMA_ENGINE_DIR overrides; otherwise default to the sibling
        // `../rullama-framework/engine` checkout. Only used when it exists.
        let engine_dir = std::env::var_os("RULLAMA_ENGINE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                let sibling = repo_root.join("../rullama-framework/engine");
                sibling.is_dir().then_some(sibling)
            })
            .filter(|p| p.is_dir());
        Ok(Paths {
            repo_root,
            pkg_dir,
            ollama_models,
            engine_dir,
        })
    }

    pub fn web_dir(&self) -> PathBuf {
        self.repo_root.join("web")
    }
    pub fn manifests_dir(&self) -> PathBuf {
        self.ollama_models.join("manifests")
    }
    pub fn blobs_dir(&self) -> PathBuf {
        self.ollama_models.join("blobs")
    }
    /// Engine source dirs to watch for cross-repo dev rebuilds. Empty when no
    /// engine checkout is present (prebuilt `pkg/` mode).
    pub fn rust_watch_dirs(&self) -> Vec<PathBuf> {
        match &self.engine_dir {
            Some(engine) => vec![
                engine.join("rullama-engine/src"),
                engine.join("rullama-lora/src"),
            ],
            None => Vec::new(),
        }
    }
}

fn find_repo_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let mut cur: &Path = &cwd;
    loop {
        // Workspace root has Cargo.toml AND a `crates/` subdir AND the
        // `web/` PWA project we serve.
        if cur.join("Cargo.toml").is_file()
            && cur.join("crates").is_dir()
            && cur.join("web").is_dir()
        {
            return Ok(cur.to_path_buf());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => {
                return Err(format!(
                    "could not find repo root by walking up from {} (expected Cargo.toml + crates/ + web/)",
                    cwd.display()
                )
                .into());
            }
        }
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub struct AppState {
    pub paths: Paths,
    pub vite_port: u16,
    pub events: broadcast::Sender<DevEvent>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum DevEvent {
    /// fs watcher saw changes and is about to kick wasm-pack. Lets the
    /// browser dim the page so the user knows a rebuild is in flight.
    WasmBuilding,
    /// wasm-pack exited 0. Page will reload after this.
    WasmRebuilt { at_ms: u128 },
    /// wasm-pack failed. The page should surface the stderr tail as a
    /// banner instead of silently serving stale bytes.
    WasmFailed { stderr_tail: String },
}
