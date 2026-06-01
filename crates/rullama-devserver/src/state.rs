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
        Ok(Paths { repo_root, pkg_dir, ollama_models })
    }

    pub fn web_dir(&self) -> PathBuf { self.repo_root.join("examples/web") }
    pub fn manifests_dir(&self) -> PathBuf { self.ollama_models.join("manifests") }
    pub fn blobs_dir(&self) -> PathBuf { self.ollama_models.join("blobs") }
    pub fn rust_watch_dirs(&self) -> Vec<PathBuf> {
        vec![
            self.repo_root.join("crates/rullama/src"),
            self.repo_root.join("crates/rullama-finetune/src"),
        ]
    }
}

fn find_repo_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    let mut cur: &Path = &cwd;
    loop {
        // Workspace root has Cargo.toml AND a `crates/` subdir AND a
        // `pkg/` subdir we expect to serve.
        if cur.join("Cargo.toml").is_file()
            && cur.join("crates").is_dir()
            && cur.join("examples").is_dir()
        {
            return Ok(cur.to_path_buf());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => {
                return Err(format!(
                    "could not find repo root by walking up from {} (expected Cargo.toml + crates/ + examples/)",
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
