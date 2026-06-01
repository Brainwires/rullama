//! Reverse-proxy fallback to Vite for any non-/api/, non-/pkg/, non-/__rullama-dev-ws
//! request. HTTP-only; Vite's HMR WebSocket connects to :5173 directly by
//! configuring `server.hmr.clientPort=5173` on the Vite side, so we don't
//! need to forward WS upgrades through axum.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{StatusCode, Uri, uri::PathAndQuery};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;

use crate::state::AppState;

pub async fn fallback_handler(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
) -> Response {
    // Build the upstream URI: http://127.0.0.1:<vite_port><path?query>.
    let pq: PathAndQuery = req
        .uri()
        .path_and_query()
        .cloned()
        .unwrap_or_else(|| PathAndQuery::from_static("/"));
    let upstream = match Uri::builder()
        .scheme("http")
        .authority(format!("127.0.0.1:{}", state.vite_port))
        .path_and_query(pq)
        .build()
    {
        Ok(u) => u,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("proxy: build uri failed: {e}"),
            )
                .into_response();
        }
    };
    *req.uri_mut() = upstream;

    // Strip hop-by-hop headers per RFC 7230 §6.1 so we don't confuse Vite.
    let h = req.headers_mut();
    for name in HOP_BY_HOP { h.remove(*name); }
    h.remove("host"); // hyper will add the correct Host for the upstream

    // Fire the request via hyper-util's legacy client.
    let client: Client<HttpConnector, Body> = Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(8)
        .build_http();

    let resp = match client.request(req).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!(
                    "proxy: upstream unreachable at 127.0.0.1:{} — is Vite up? ({e})",
                    state.vite_port
                ),
            )
                .into_response();
        }
    };

    // Re-construct the axum Response with the upstream body buffered →
    // streamed. For dev usage the bodies are small (HTML/JS chunks).
    let (parts, body) = resp.into_parts();
    let bytes = match body.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("proxy: read upstream body: {e}"),
            )
                .into_response();
        }
    };
    let mut builder = Response::builder().status(parts.status);
    for (k, v) in parts.headers.iter() {
        if HOP_BY_HOP.iter().any(|n| n.eq_ignore_ascii_case(k.as_str())) {
            continue;
        }
        builder = builder.header(k, v);
    }
    builder.body(Body::from(bytes)).unwrap()
}

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];
