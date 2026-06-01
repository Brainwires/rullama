//! Runtime configuration knobs for the devserver. Composed from CLI args
//! in `bin/server.rs`; the most important is `Mode::Public`, which
//! switches the safe defaults for tunnel-exposed operation.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Local dev — Vite reverse proxy, /api/log writable, /api/models
    /// listed, /__rullama-dev-ws broadcasting. Convenient but leaks the
    /// whole repo via Vite if any public origin can reach :PORT.
    #[default]
    LocalDev,
    /// Tunnel-safe — Vite proxy replaced by a static serve of
    /// `examples/web/dist/`, model list disabled, log write disabled,
    /// dev WS disabled. The mode you want when cloudflared is up.
    Public,
}

#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub mode: Mode,
    pub allow_models: bool,
    pub allow_log_write: bool,
    pub allow_dev_ws: bool,
    pub serve_dist: bool,
    /// Allowed `Origin` values for CORS. Empty list = no cross-origin
    /// access. `*` is NEVER allowed (we never wildcard).
    pub cors_origins: Vec<String>,
    /// Soft cap on /api/log body bytes. Bigger requests are rejected
    /// with 413 before they hit disk.
    pub api_log_max_bytes: usize,
    /// Where `/api/log` writes. Tests inject a per-fixture path so
    /// parallel runs don't collide. `None` → defer to env var
    /// `RULLAMA_PAGE_LOG` then default `/tmp/rullama-page.log`.
    pub api_log_path: Option<std::path::PathBuf>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            mode: Mode::LocalDev,
            allow_models: true,
            allow_log_write: true,
            allow_dev_ws: true,
            serve_dist: false,
            cors_origins: vec![],
            api_log_max_bytes: 8 * 1024,
            api_log_path: None,
        }
    }
}

impl SecurityConfig {
    /// Build the safe-defaults preset for public hosting. The caller can
    /// still pass `--allow-models` etc. to flip individual knobs back on.
    pub fn public_defaults() -> Self {
        Self {
            mode: Mode::Public,
            allow_models: false,
            allow_log_write: false,
            allow_dev_ws: false,
            serve_dist: true,
            cors_origins: vec![],
            api_log_max_bytes: 4 * 1024,
            api_log_path: None,
        }
    }

    pub fn allow_origin(&self, origin: &str) -> bool {
        self.cors_origins.iter().any(|o| o == origin)
    }
}
