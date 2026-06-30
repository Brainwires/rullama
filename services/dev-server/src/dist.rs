//! Public-safe fallback: serve `web/dist/` directly off disk,
//! with the same SPA-fallback semantics serve-tunnel.sh has — paths that
//! don't resolve to an existing file AND don't look like an asset path
//! (no extension) get the `dist/index.html` shell, so client-side
//! routing still works.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::stream;
use tokio::io::AsyncReadExt;

use crate::state::AppState;

const CHUNK: usize = 64 * 1024;

pub async fn fallback_handler(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let raw = req.uri().path().trim_start_matches('/');
    // Browsers percent-encode spaces and other reserved chars; the
    // on-disk filename has the raw byte. Decode before joining or any
    // `icons/icon-192 - back.png`-style filename 404s with a literal
    // `%20` lookup. Decoding also normalises `+`-as-space (some legacy
    // clients), but here we keep the raw `+` since path segments don't
    // canonically use it as a space.
    let decoded = percent_decode(raw);
    let path: &str = &decoded;
    let dist_dir = state.paths.repo_root.join("apps/web/dist");
    // Canonicalize the base ONCE and reject any path that escapes it.
    let base = match tokio::fs::canonicalize(&dist_dir).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "dist/ not built; run `pnpm --filter rullama-web build` (looked in {})",
                    dist_dir.display()
                ),
            )
                .into_response();
        }
    };
    let candidate = if path.is_empty() {
        base.join("index.html")
    } else {
        base.join(path)
    };
    let resolved = match tokio::fs::canonicalize(&candidate).await {
        Ok(r) => r,
        Err(_) => {
            // SPA fallback: missing file AND no extension → serve index.html.
            if path
                .split('/')
                .next_back()
                .map(|p| !p.contains('.'))
                .unwrap_or(true)
            {
                match tokio::fs::canonicalize(base.join("index.html")).await {
                    Ok(r) => r,
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("dist/index.html: {e}"),
                        )
                            .into_response();
                    }
                }
            } else {
                return StatusCode::NOT_FOUND.into_response();
            }
        }
    };
    if !resolved.starts_with(&base) {
        return StatusCode::FORBIDDEN.into_response();
    }
    serve_file(resolved).await
}

async fn serve_file(path: PathBuf) -> Response {
    let meta = match tokio::fs::metadata(&path).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
    };
    if !meta.is_file() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let len = meta.len();
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
    // /dist/index.html, JS, CSS, wasm — same no-store policy we use for
    // /pkg/, so a fresh deploy is picked up on next page load.
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if matches!(ext, "js" | "html" | "wasm" | "css" | "mjs" | "map" | "json") {
        resp.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store, no-cache, must-revalidate, max-age=0"),
        );
    } else {
        // Icons / images — let the browser cache for an hour.
        resp.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("public, max-age=3600"),
        );
    }
    resp
}

/// Minimal percent-decode for URL path segments. Returns the original
/// string borrow when there's nothing to decode (no `%`).
fn percent_decode(input: &str) -> String {
    if !input.contains('%') {
        return input.to_string();
    }
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_nibble(bytes[i + 1]);
            let lo = hex_nibble(bytes[i + 2]);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn mime_for(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "wasm" => "application/wasm",
        "js" | "mjs" => "application/javascript",
        "css" => "text/css; charset=utf-8",
        "html" => "text/html; charset=utf-8",
        "json" | "map" => "application/json",
        // Spec-correct type so iOS/Safari unambiguously treats it as a PWA
        // manifest (it accepts application/json too, but this removes all doubt).
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}
