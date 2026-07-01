//! /api/* routes — bit-identical to `web/serve-tunnel.sh`'s wire
//! shape so the PWA and `mac-cdp-test.mjs` don't need any client-side change.
//!
//! Endpoints:
//!   GET/HEAD /api/models            — Ollama model list
//!   GET/HEAD /api/blob/{family}:{tag} — Range-streamed GGUF blob
//!   POST     /api/log               — append `[<tag>] <msg>\n` to /tmp/rullama-page.log
//!   OPTIONS  /api/*                  — CORS preflight (204)

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_util::stream;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};

use crate::state::AppState;

const MODEL_LAYER_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const DEFAULT_PAGE_LOG: &str = "/tmp/rullama-page.log";
const CHUNK: usize = 1 << 20;

fn page_log_path() -> std::path::PathBuf {
    std::env::var_os("RULLAMA_PAGE_LOG")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_PAGE_LOG))
}

/// Backwards-compat: all three /api/* routes mounted on one Router.
/// Used by tests + `build_app(.., SecurityConfig::default())`.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(models_router())
        .merge(blob_router())
        .merge(log_router())
}

pub fn models_router() -> Router<Arc<AppState>> {
    Router::new().route(
        "/api/models",
        get(get_models).head(head_models).options(options_204),
    )
}

pub fn blob_router() -> Router<Arc<AppState>> {
    Router::new().route(
        "/api/blob/*key",
        get(get_blob).head(head_blob).options(options_204),
    )
}

pub fn log_router() -> Router<Arc<AppState>> {
    Router::new().route("/api/log", post(post_log).options(options_204))
}

async fn options_204() -> impl IntoResponse {
    StatusCode::NO_CONTENT
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub family: String,
    pub tag: String,
    pub size: u64,
    pub digest: String,
    pub filename: String,
    #[serde(rename = "modelKey")]
    pub model_key: String,
    pub multimodal: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaManifestLayer {
    #[serde(default, rename = "mediaType")]
    media_type: String,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    layers: Vec<OllamaManifestLayer>,
}

async fn discover_models(state: &AppState) -> Vec<ModelEntry> {
    let manifests = state.paths.manifests_dir();
    let blobs = state.paths.blobs_dir();
    let mut out: Vec<ModelEntry> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![manifests.clone()];
    while let Some(dir) = stack.pop() {
        let mut rd = match fs::read_dir(&dir).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            let ft = match entry.file_type().await {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push(path);
                continue;
            }
            // Treat any plain file under manifests/ as a candidate (no .json
            // suffix on disk for newer Ollama versions either).
            let bytes = match fs::read(&path).await {
                Ok(b) => b,
                Err(_) => continue,
            };
            let manifest: OllamaManifest = match serde_json::from_slice(&bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };
            // family/tag = the last two path segments under manifests/ —
            // e.g. registry.ollama.ai/library/gemma4/e2b → family=gemma4 tag=e2b.
            let rel = match path.strip_prefix(&manifests) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let parts: Vec<_> = rel.components().collect();
            if parts.len() < 2 {
                continue;
            }
            let tag = parts[parts.len() - 1]
                .as_os_str()
                .to_string_lossy()
                .to_string();
            let family = parts[parts.len() - 2]
                .as_os_str()
                .to_string_lossy()
                .to_string();
            for layer in manifest.layers.iter() {
                if layer.media_type != MODEL_LAYER_MEDIA_TYPE {
                    continue;
                }
                let digest = layer
                    .digest
                    .strip_prefix("sha256:")
                    .unwrap_or(&layer.digest)
                    .to_string();
                let blob_path = blobs.join(format!("sha256-{digest}"));
                let blob_size = match fs::metadata(&blob_path).await {
                    Ok(m) => m.len(),
                    Err(_) => continue,
                };
                let size = if layer.size > 0 {
                    layer.size
                } else {
                    blob_size
                };
                let name = format!("{family}:{tag}");
                out.push(ModelEntry {
                    name: name.clone(),
                    family: family.clone(),
                    tag: tag.clone(),
                    size,
                    digest: digest.clone(),
                    filename: format!("sha256-{digest}"),
                    model_key: name,
                    multimodal: false,
                });
                break;
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

async fn get_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models = discover_models(&state).await;
    Json(models)
}

async fn head_models() -> impl IntoResponse {
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json; charset=utf-8")
        .body(Body::empty())
        .unwrap();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    resp
}

async fn head_blob(
    State(state): State<Arc<AppState>>,
    AxumPath(key): AxumPath<String>,
) -> Response {
    let key = urlencoding_decode(&key);
    let Some(blob) = find_blob_path(&state, &key).await else {
        return (StatusCode::NOT_FOUND, format!("model not found: {key}")).into_response();
    };
    let meta = match fs::metadata(&blob).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
    };
    let total = meta.len();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, total.to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .header("X-Model-Name", &key)
        .header("X-Total-Size", total.to_string())
        .body(Body::empty())
        .unwrap()
}

async fn get_blob(
    State(state): State<Arc<AppState>>,
    AxumPath(key): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    let key = urlencoding_decode(&key);
    let Some(blob) = find_blob_path(&state, &key).await else {
        return (StatusCode::NOT_FOUND, format!("model not found: {key}")).into_response();
    };
    let meta = match fs::metadata(&blob).await {
        Ok(m) => m,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
    };
    let size = meta.len();

    // Parse Range: bytes=<start>-<end> (either bound optional).
    let (start, end, partial) = match parse_range(&headers, size) {
        Some(r) => r,
        None => (0u64, size.saturating_sub(1), false),
    };
    let length = end + 1 - start;

    let mut file = match tokio::fs::File::open(&blob).await {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("open: {e}")).into_response(),
    };
    if let Err(e) = file.seek(SeekFrom::Start(start)).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("seek: {e}")).into_response();
    }

    // Stream in 1 MiB chunks. We bound by `remaining` so we don't over-read.
    let mut remaining = length;
    let body_stream = stream::unfold((file, remaining), |(mut f, mut rem)| async move {
        if rem == 0 {
            return None;
        }
        let want = rem.min(CHUNK as u64) as usize;
        let mut buf = vec![0u8; want];
        match f.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                rem -= n as u64;
                Some((Ok::<_, std::io::Error>(Bytes::from(buf)), (f, rem)))
            }
            Err(_) => None,
        }
    });
    let _ = &mut remaining; // silence move warning
    let body = Body::from_stream(body_stream);

    let status = if partial {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, length.to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .header("X-Model-Name", &key)
        .header("X-Total-Size", size.to_string());
    if partial {
        builder = builder.header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{size}"));
    }
    builder.body(body).unwrap()
}

fn parse_range(headers: &HeaderMap, size: u64) -> Option<(u64, u64, bool)> {
    let h = headers.get(header::RANGE)?.to_str().ok()?;
    let spec = h.strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    let start: u64 = if a.is_empty() { 0 } else { a.parse().ok()? };
    let end: u64 = if b.is_empty() {
        size.saturating_sub(1)
    } else {
        b.parse().ok()?
    };
    let end = end.min(size.saturating_sub(1));
    if size == 0 || start > end || start >= size {
        return None;
    }
    Some((start, end, true))
}

async fn find_blob_path(state: &AppState, name_tag: &str) -> Option<PathBuf> {
    // Kokoro TTS GGUF is not an Ollama model — serve the local file so the PWA can
    // pull it via ?localBlob (bypassing the R2 CDN). Override path via KOKORO_GGUF.
    if name_tag == "kokoro:82m" {
        let p = std::env::var("KOKORO_GGUF")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default())
                    .join(".cache/kokoro/kokoro-82m-f16.gguf")
            });
        return p.is_file().then_some(p);
    }
    // StyleTTS2-LibriTTS voice-cloning GGUF (f32). Override path via STYLETTS2_GGUF.
    if name_tag == "styletts2:libritts" {
        let p = std::env::var("STYLETTS2_GGUF")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default())
                    .join(".cache/styletts2/styletts2-libritts-f32.gguf")
            });
        return p.is_file().then_some(p);
    }
    let models = discover_models(state).await;
    let m = models.into_iter().find(|m| m.name == name_tag)?;
    Some(state.paths.blobs_dir().join(format!("sha256-{}", m.digest)))
}

#[derive(Debug, Deserialize)]
struct LogPayload {
    #[serde(default)]
    tag: String,
    #[serde(default)]
    msg: String,
}

async fn post_log(
    axum::extract::Extension(cfg): axum::extract::Extension<crate::config::SecurityConfig>,
    Json(payload): Json<LogPayload>,
) -> Response {
    let tag = if payload.tag.is_empty() {
        "?".to_string()
    } else {
        payload.tag
    };
    let line = format!("[{tag}] {}\n", payload.msg);
    let log_path = cfg.api_log_path.clone().unwrap_or_else(page_log_path);
    let res = async {
        let mut f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .await?;
        f.write_all(line.as_bytes()).await?;
        // Tokio's File doesn't flush on drop. Without this, parallel
        // test runs read the file before bytes hit disk and see empty.
        f.flush().await?;
        Ok::<_, std::io::Error>(())
    }
    .await;
    match res {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("log fail: {e}")).into_response(),
    }
}

// We don't need full urlencoding for the model-key shape, but `:` and a
// few others are common. axum's *catch-all path stays %-encoded; decode it.
fn urlencoding_decode(s: &str) -> String {
    // Minimal: percent-decode bytes; non-percent chars pass through.
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hexval(bytes[i + 1]);
            let lo = hexval(bytes[i + 2]);
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

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}
