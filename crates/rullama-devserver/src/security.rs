//! Security headers + CORS allow-list + per-route hardening.
//!
//! All responses pick up `Cross-Origin-Opener-Policy: same-origin` and
//! `Cross-Origin-Embedder-Policy: require-corp` (the cross-origin-isolated
//! context the PWA's SharedArrayBuffer / WebGPU paths assume).
//!
//! `Cross-Origin-Resource-Policy` is `cross-origin` for `/api/blob` and
//! `/api/models` (so the page on `rullama.brainwires.net` can fetch them
//! from `http://localhost:25321` via the `?localBlob=` flow); everything
//! else gets `same-origin`.
//!
//! `Access-Control-Allow-Origin` is set ONLY when the request's `Origin`
//! is in `SecurityConfig::cors_origins`. Public mode defaults to an empty
//! allow-list; dev mode defaults to permissive echo.

use axum::http::{HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::Response;
use axum::extract::Request;

use crate::config::SecurityConfig;

/// Tower middleware: COOP/COEP/CORP + conditional CORS echo + always
/// `Vary: Origin`. Lives at the Router root so it runs after every
/// handler.
pub async fn apply_security_headers(
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let method = req.method().clone();
    let cfg: SecurityConfig = req
        .extensions()
        .get::<SecurityConfig>()
        .cloned()
        .unwrap_or_default();
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    // COOP/COEP — harmless on subresources; mandatory on the top-level
    // document for the cross-origin-isolated context the PWA needs.
    h.insert(
        "cross-origin-opener-policy",
        HeaderValue::from_static("same-origin"),
    );
    h.insert(
        "cross-origin-embedder-policy",
        HeaderValue::from_static("require-corp"),
    );
    // CORP — pick cross-origin for the two cross-origin-fetch routes.
    let corp = if path.starts_with("/api/blob/") || path == "/api/models" || path.starts_with("/pkg/") {
        "cross-origin"
    } else {
        "same-origin"
    };
    h.insert("cross-origin-resource-policy", HeaderValue::from_static(corp));
    // CORS — only echo allowed origins. `Vary: Origin` so caches don't
    // serve the wrong allow-origin to the wrong client.
    h.insert(header::VARY, HeaderValue::from_static("Origin"));
    if let Some(origin) = origin {
        if cfg.allow_origin(&origin) {
            if let Ok(v) = HeaderValue::from_str(&origin) {
                h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
            }
            h.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                HeaderValue::from_static("Range, Content-Type"),
            );
            h.insert(
                header::ACCESS_CONTROL_EXPOSE_HEADERS,
                HeaderValue::from_static(
                    "Content-Length, Content-Range, Accept-Ranges, X-Model-Name, X-Total-Size",
                ),
            );
            h.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                HeaderValue::from_static("GET, HEAD, OPTIONS, POST"),
            );
        }
    }
    // Mute the OPTIONS preflight body — already 204 from the handler.
    if method == Method::OPTIONS {
        // no-op; body is empty
    }
    resp
}
