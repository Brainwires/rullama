//! `/api/cloud/{provider}/*` — BYOK reverse-proxy to Ollama Cloud / OpenAI.
//!
//! Both upstreams send **no CORS headers**, so a browser can't call them
//! directly; this route is the mandatory server-side hop. The browser sends
//! the user's key in an `X-Cloud-Key` header (NOT `Authorization`, so the
//! browser's own fetch never attaches ambient credentials); we set
//! `Authorization: Bearer <key>` on the upstream request and stream the
//! (Server-Sent-Events) response straight back, unbuffered. The key is
//! **never logged, persisted, or echoed**.
//!
//! This is the **dev / `cargo dev` path only**. In Docker/production the PWA
//! hits the same same-origin `/api/cloud/*` path but nginx reverse-proxies it
//! to a Cloudflare Worker (see `docker/nginx-rullama.conf`), so the residential
//! host IP never makes the upstream call.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::Path as AxumPath;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;

use crate::state::AppState;

/// Header the browser carries the BYOK key in. Lower-case for `HeaderMap`
/// lookups (header names are case-insensitive but the map keys are lower).
const CLOUD_KEY_HEADER: &str = "x-cloud-key";

/// Map a provider slug to its OpenAI-compatible base URL. Both expose
/// `/chat/completions` and `/models` with identical request/response shapes.
fn upstream_base(provider: &str) -> Option<&'static str> {
    match provider {
        "ollama" => Some("https://ollama.com/v1"),
        "openai" => Some("https://api.openai.com/v1"),
        _ => None,
    }
}

pub fn cloud_router() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/cloud/:provider/chat",
            post(post_chat).options(options_204),
        )
        .route(
            "/api/cloud/:provider/models",
            get(get_models).options(options_204),
        )
}

async fn options_204() -> impl IntoResponse {
    StatusCode::NO_CONTENT
}

/// POST `/api/cloud/{provider}/chat` → upstream `{base}/chat/completions`.
/// Streams the response body verbatim (SSE when the client asked for
/// `stream:true`). The request JSON body is forwarded as-is — the browser
/// already shapes the OpenAI-compatible payload.
async fn post_chat(AxumPath(provider): AxumPath<String>, headers: HeaderMap, body: Bytes) -> Response {
    let Some(base) = upstream_base(&provider) else {
        return (StatusCode::BAD_REQUEST, format!("unknown provider: {provider}")).into_response();
    };
    let key = match cloud_key(&headers) {
        Some(k) => k,
        None => return (StatusCode::UNAUTHORIZED, "missing X-Cloud-Key").into_response(),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(body)
        .send()
        .await;
    relay(resp, "text/event-stream")
}

/// GET `/api/cloud/{provider}/models` → upstream `{base}/models`. Small JSON;
/// still streamed for code uniformity.
async fn get_models(AxumPath(provider): AxumPath<String>, headers: HeaderMap) -> Response {
    let Some(base) = upstream_base(&provider) else {
        return (StatusCode::BAD_REQUEST, format!("unknown provider: {provider}")).into_response();
    };
    let key = match cloud_key(&headers) {
        Some(k) => k,
        None => return (StatusCode::UNAUTHORIZED, "missing X-Cloud-Key").into_response(),
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base}/models"))
        .header("authorization", format!("Bearer {key}"))
        .send()
        .await;
    relay(resp, "application/json")
}

/// Pull the BYOK key out of the request, rejecting an empty value. Returns
/// the raw key string — it is used immediately and never stored.
fn cloud_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get(CLOUD_KEY_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string)
}

/// Turn an upstream `reqwest` result into an axum streaming response. The
/// upstream status and Content-Type are relayed; the body streams through
/// unbuffered (`Body::from_stream`) so SSE deltas arrive incrementally — and
/// non-2xx error bodies (e.g. `{"error":{...}}` for a bad key) pass through
/// verbatim. A transport failure maps to 502.
fn relay(resp: Result<reqwest::Response, reqwest::Error>, default_ct: &str) -> Response {
    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            // Deliberately do NOT include the key or full request in the error.
            return (StatusCode::BAD_GATEWAY, format!("cloud upstream unreachable: {e}"))
                .into_response();
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(default_ct)
        .to_string();
    let body = Body::from_stream(resp.bytes_stream());
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, ct)
        .header(header::CACHE_CONTROL, "no-cache")
        // Defeat any intermediary buffering so SSE tokens are not held back.
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}
