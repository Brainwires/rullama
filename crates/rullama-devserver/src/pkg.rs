//! /pkg/* static routes.
//!
//! Serves the wasm-pack output directly off `<repo>/pkg/*` with
//! `Cache-Control: no-store` so a fresh wasm rebuild is picked up on the
//! next page load with zero browser-cache interference.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use bytes::Bytes;
use futures_util::stream;
use tokio::io::AsyncReadExt;

use crate::state::AppState;

const CHUNK: usize = 64 * 1024;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/pkg/*path", get(serve_pkg).head(head_pkg))
}

async fn resolve_safe(state: &AppState, sub: &str) -> Option<PathBuf> {
    // Component-level reject (cheap, catches plain `..`).
    if sub.split('/').any(|p| p == ".." || p.is_empty()) {
        return None;
    }
    let candidate = state.paths.pkg_dir.join(sub);
    if !tokio::fs::metadata(&candidate).await.ok()?.is_file() {
        return None;
    }
    // Canonicalization-level reject — catches symlink escapes, encoded
    // traversal that decoded into ".." after our component check
    // (shouldn't happen since we run on the already-decoded String, but
    // it's defense-in-depth), and any future bug in the component check.
    let canon = tokio::fs::canonicalize(&candidate).await.ok()?;
    let base = tokio::fs::canonicalize(&state.paths.pkg_dir).await.ok()?;
    if !canon.starts_with(&base) {
        return None;
    }
    Some(canon)
}

async fn head_pkg(State(state): State<Arc<AppState>>, AxumPath(sub): AxumPath<String>) -> Response {
    let Some(path) = resolve_safe(&state, &sub).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let len = match tokio::fs::metadata(&path).await {
        Ok(m) => m.len(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
    };
    let mime = mime_for(&path);
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .header(header::CONTENT_LENGTH, len.to_string())
        .body(Body::empty())
        .unwrap();
    add_no_store_for_assets(&path, resp.headers_mut());
    resp
}

async fn serve_pkg(
    State(state): State<Arc<AppState>>,
    AxumPath(sub): AxumPath<String>,
) -> Response {
    let Some(path) = resolve_safe(&state, &sub).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let len = match tokio::fs::metadata(&path).await {
        Ok(m) => m.len(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
    };
    let file = match tokio::fs::File::open(&path).await {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("open: {e}")).into_response(),
    };
    let body_stream = stream::unfold(file, |mut f| async move {
        let mut buf = vec![0u8; CHUNK];
        match f.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok::<_, std::io::Error>(Bytes::from(buf)), f))
            }
            Err(_) => None,
        }
    });
    let mime = mime_for(&path);
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime)
        .header(header::CONTENT_LENGTH, len.to_string())
        .body(Body::from_stream(body_stream))
        .unwrap();
    add_no_store_for_assets(&path, resp.headers_mut());
    resp
}

fn mime_for(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "wasm" => "application/wasm",
        "js" | "mjs" => "application/javascript",
        "css" => "text/css; charset=utf-8",
        "html" => "text/html; charset=utf-8",
        "map" => "application/json",
        "ts" => "application/typescript",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

fn add_no_store_for_assets(path: &std::path::Path, headers: &mut axum::http::HeaderMap) {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if matches!(ext, "js" | "html" | "wasm" | "ts" | "css" | "mjs" | "map") {
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store, no-cache, must-revalidate, max-age=0"),
        );
        headers.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
        headers.insert(header::EXPIRES, HeaderValue::from_static("0"));
    }
}
