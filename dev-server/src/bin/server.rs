//! rullama-devserver binary entry — single-command local dev server.
//!
//! See `dev-server/src/lib.rs` for the router builder; this
//! file is just orchestration: arg parsing, fail-fast port check, Vite
//! child spawn, watcher spawn, axum bind, signal handling.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tokio::sync::broadcast;
use tracing_subscriber::{EnvFilter, fmt};

use rullama_devserver::{AppState, DevEvent, Paths, SecurityConfig, build_app};
use rullama_devserver::{vite, watcher};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "rullama-devserver",
    about = "Local dev server: /api/* + /pkg/* + Vite reverse-proxy + Rust → WASM watch.\n\nFor tunneled (Cloudflare) public hosting use `--public`."
)]
struct Args {
    #[arg(long, default_value_t = 25321)]
    port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 5173)]
    vite_port: u16,
    #[arg(long)]
    no_watch: bool,
    #[arg(long)]
    no_vite: bool,
    #[arg(long)]
    ollama_models: Option<PathBuf>,
    #[arg(long)]
    repo_root: Option<PathBuf>,

    // --- security knobs (see dev-server/src/config.rs) ---
    /// Apply tunnel-safe defaults: serve dist/ instead of Vite proxy,
    /// disable /api/log, /api/models, /__rullama-dev-ws. Use this when
    /// cloudflared is up and the devserver is exposed at a public URL.
    #[arg(long)]
    public: bool,

    /// Serve `web/dist/*` as the SPA root instead of reverse-
    /// proxying to Vite. Implied by `--public`.
    #[arg(long)]
    serve_dist: bool,

    /// Disable Vite reverse-proxy fallback (404 instead). Useful for
    /// `--public` + a separately-hosted dist via CDN, or for hermetic
    /// tests.
    #[arg(long)]
    no_proxy: bool,

    /// Explicit CORS origin allow-list (repeatable). Default for public
    /// mode: empty (no cross-origin reads). Default for local-dev: empty
    /// (relies on same-origin; pass --cors-origins to enable).
    #[arg(long, value_delimiter = ',')]
    cors_origins: Vec<String>,

    /// Override individual `--public` defaults if you really want to.
    #[arg(long)]
    allow_models: bool,
    #[arg(long)]
    allow_log_write: bool,
    #[arg(long)]
    allow_dev_ws: bool,

    /// Disable the BYOK cloud proxy at `/api/cloud/*` (it is ON by default
    /// in both local-dev and `--public`).
    #[arg(long)]
    no_cloud: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("rullama_devserver=info,tower_http=info,axum=info,warn")
        }))
        .with_target(false)
        .init();

    let args = Args::parse();
    let paths = Paths::resolve(args.repo_root.clone(), args.ollama_models.clone())?;
    tracing::info!("repo_root      = {}", paths.repo_root.display());
    tracing::info!("pkg_dir        = {}", paths.pkg_dir.display());
    tracing::info!("ollama_models  = {}", paths.ollama_models.display());
    tracing::info!("vite_upstream  = http://127.0.0.1:{}", args.vite_port);

    let bind_addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    if let Err(e) = std::net::TcpListener::bind(bind_addr) {
        eprintln!();
        eprintln!("✗ port {} already in use: {}", args.port, e);
        eprintln!("  Likely the Python serve-tunnel.sh is still running. To free the port:");
        eprintln!("    lsof -nP -iTCP:{} -sTCP:LISTEN", args.port);
        eprintln!("    kill <pid>");
        eprintln!("  Then re-run `cargo dev`.");
        eprintln!(
            "  Do NOT kill cloudflared — it auto-reconnects to whatever's at :{}.",
            args.port
        );
        std::process::exit(1);
    }

    // Compose SecurityConfig from the safe-defaults preset (if --public)
    // or the permissive local-dev defaults, then layer the explicit flags
    // on top — so `--public --allow-models` is "tunnel-safe except models
    // is back on" without making the user remember every individual flag.
    let mut cfg = if args.public {
        SecurityConfig::public_defaults()
    } else {
        SecurityConfig::default()
    };
    if args.serve_dist {
        cfg.serve_dist = true;
    }
    if args.no_proxy {
        cfg.serve_dist = true;
    } // no proxy ≈ static-only
    if args.allow_models {
        cfg.allow_models = true;
    }
    if args.allow_log_write {
        cfg.allow_log_write = true;
    }
    if args.allow_dev_ws {
        cfg.allow_dev_ws = true;
    }
    if args.no_cloud {
        cfg.allow_cloud = false;
    }
    if !args.cors_origins.is_empty() {
        cfg.cors_origins = args.cors_origins.clone();
    }
    tracing::info!("mode           = {:?}", cfg.mode);
    tracing::info!("serve_dist     = {}", cfg.serve_dist);
    tracing::info!("allow_models   = {}", cfg.allow_models);
    tracing::info!("allow_log      = {}", cfg.allow_log_write);
    tracing::info!("allow_dev_ws   = {}", cfg.allow_dev_ws);
    tracing::info!("allow_cloud    = {}", cfg.allow_cloud);
    tracing::info!("cors_origins   = {:?}", cfg.cors_origins);

    let (tx, _rx0) = broadcast::channel::<DevEvent>(64);
    let state = Arc::new(AppState {
        paths: paths.clone(),
        vite_port: args.vite_port,
        events: tx.clone(),
    });
    let app = build_app(state, cfg.clone());

    // Spawn Vite only if we're actually going to proxy to it.
    let want_vite = !args.no_vite && !cfg.serve_dist;
    let vite_handle = if want_vite {
        Some(vite::spawn_vite(paths.web_dir(), args.vite_port).await?)
    } else {
        None
    };
    let watcher_handle = if args.no_watch {
        None
    } else {
        Some(watcher::spawn(paths.clone(), tx.clone()))
    };

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let shutdown_tx_handler = shutdown_tx.clone();
    ctrlc::set_handler(move || {
        let _ = shutdown_tx_handler.try_send(());
    })?;

    tracing::info!("listening on http://{}", bind_addr);
    tracing::info!(
        "  → React dev (Vite HMR) reverse-proxied from :{}",
        args.vite_port
    );
    tracing::info!("  → /api/* served natively (parity with serve-tunnel.sh)");
    tracing::info!("  → /pkg/* served from {}", paths.pkg_dir.display());
    tracing::info!("  → WS broadcast at /__rullama-dev-ws");

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = shutdown_rx.recv().await;
        tracing::info!("shutting down…");
    });

    if let Err(e) = server.await {
        tracing::error!("axum serve error: {e}");
    }

    if let Some(mut h) = vite_handle {
        h.shutdown().await;
    }
    if let Some(h) = watcher_handle {
        drop(h);
    }
    Ok(())
}
