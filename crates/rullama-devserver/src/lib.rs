//! rullama-devserver library — exposes the router builder and shared
//! types so tests can drive the server without binding a TCP port.
//!
//! See `src/bin/server.rs` for the binary entry point.

pub mod api;
pub mod config;
pub mod dist;
pub mod pkg;
pub mod proxy;
pub mod security;
pub mod state;
pub mod vite;
pub mod watcher;
pub mod ws;

use std::sync::Arc;
use std::time::Duration;

use axum::Extension;
use axum::Router;
use axum::middleware;
use tokio::sync::broadcast;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

pub use config::{Mode, SecurityConfig};
pub use state::{AppState, DevEvent, Paths};

/// Build the full axum Router with the application state attached.
///
/// `cfg` toggles per-route behavior (public vs local-dev, what routes
/// are mounted at all). Tests should pass `SecurityConfig::default()`
/// for the permissive local-dev shape.
pub fn build_app(state: Arc<AppState>, cfg: SecurityConfig) -> Router {
    let mut app = Router::new();
    if cfg.allow_models {
        app = app.merge(api::models_router());
    }
    if cfg.allow_log_write {
        // /api/log gets a tight body limit so a hostile client can't
        // fill the user's disk with a single oversized POST.
        let log = api::log_router()
            .layer(RequestBodyLimitLayer::new(cfg.api_log_max_bytes));
        app = app.merge(log);
    }
    // /api/blob is mounted in BOTH modes — without it the PWA can't
    // pull GGUF blobs at all, which defeats the point of hosting it.
    // The bandwidth-amplification concern is addressed via
    // Cloudflare-side rate limiting (see CLAUDE.md), not in code.
    app = app.merge(api::blob_router());
    app = app.merge(pkg::router());
    if cfg.allow_dev_ws {
        app = app.merge(ws::router());
    }
    // Fallback: dist static serve (public) vs Vite reverse proxy (dev).
    let app = if cfg.serve_dist {
        app.fallback(dist::fallback_handler)
    } else {
        app.fallback(proxy::fallback_handler)
    };
    app
        .with_state(state)
        .layer(Extension(cfg))
        .layer(middleware::from_fn(security::apply_security_headers))
        // 30 s timeout is generous enough for the /api/blob streaming
        // case (we send chunks regularly so the inactivity timer
        // resets) while still cutting off slow-loris attempts.
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(30),
        ))
        .layer(TraceLayer::new_for_http())
}

/// Convenience builder for tests: a fresh broadcast channel + an
/// `AppState` over the supplied paths and vite port (default 5173,
/// unused unless the test exercises the proxy fallback).
pub fn test_state(paths: Paths) -> Arc<AppState> {
    let (events, _rx) = broadcast::channel::<DevEvent>(8);
    Arc::new(AppState {
        paths,
        vite_port: 5173,
        events,
    })
}
