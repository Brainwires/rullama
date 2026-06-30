//! Layer 1 integration tests — drive the axum Router in-process via
//! `tower::ServiceExt::oneshot`. No TCP port bind, no Vite, no watcher.
//!
//! Coverage: every `/api/*` and `/pkg/*` response shape declared in
//! the plan, against a synthetic Ollama tree under a tempdir so the
//! tests don't depend on the user's `~/.ollama/models`.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use tempfile::TempDir;
use tower::ServiceExt;

use rullama_devserver::{Paths, build_app, test_state};

const GGUF_MAGIC: &[u8] = b"GGUF\x03\x00\x00\x00";
const FIXTURE_BLOB_SIZE: usize = 8192;

struct Fixture {
    _tempdir: TempDir,
    paths: Paths,
    /// Pre-computed for test convenience.
    digest: String,
    blob_size: u64,
}

fn build_fixture() -> Fixture {
    let tempdir = TempDir::new().expect("create tempdir");
    let root = tempdir.path().to_path_buf();
    let models = root.join(".ollama").join("models");
    let manifests = models
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join("gemma4");
    let blobs = models.join("blobs");
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::create_dir_all(&blobs).unwrap();

    // Synthetic blob: GGUF magic + zero padding to 8 KiB so range tests
    // can verify boundary behavior.
    let mut blob = vec![0u8; FIXTURE_BLOB_SIZE];
    blob[..GGUF_MAGIC.len()].copy_from_slice(GGUF_MAGIC);

    // Compute sha256 of the blob so the manifest digest matches the file.
    // We use a minimal local sha256 — for tests we cheat with a known
    // hex string and just match it; serve-tunnel parity doesn't care
    // what the actual hash is, only that the manifest layer's digest
    // matches the on-disk file name. So generate the digest from the
    // payload via std-only via the simple recipe of writing bytes to a
    // file named with a digest, then reading the digest from the path.
    // Easiest: hash with sha2.
    use sha2_::Digest;
    let mut h = sha2_::Sha256::new();
    h.update(&blob);
    let digest = hex_encode(&h.finalize());

    let blob_path = blobs.join(format!("sha256-{}", digest));
    std::fs::write(&blob_path, &blob).unwrap();

    let manifest = serde_json::json!({
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": format!("sha256:{}", digest),
                "size": FIXTURE_BLOB_SIZE,
            }
        ]
    });
    let manifest_path = manifests.join("e2b");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    // We need a `repo_root` too because Paths::resolve auto-detects from
    // CWD. Build a minimal stub structure the resolver accepts: a dir
    // with Cargo.toml + crates/ + examples/.
    let stub = root.join("repo-stub");
    std::fs::create_dir_all(stub.join("crates")).unwrap();
    std::fs::create_dir_all(stub.join("web/dist")).unwrap();
    std::fs::create_dir_all(stub.join("pkg")).unwrap();
    std::fs::write(stub.join("Cargo.toml"), "[workspace]\n").unwrap();
    // Drop a tiny fixture file in pkg/ so /pkg/* tests have something to
    // serve.
    std::fs::write(stub.join("pkg/rullama.js"), "// fixture\n").unwrap();
    std::fs::write(stub.join("pkg/rullama_bg.wasm"), b"\x00asm\x01\x00\x00\x00").unwrap();

    let paths = Paths::resolve(Some(stub.clone()), Some(models)).unwrap();
    Fixture {
        _tempdir: tempdir,
        paths,
        digest,
        blob_size: FIXTURE_BLOB_SIZE as u64,
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0F) as usize] as char);
    }
    s
}

// We need sha2 in dev-deps. Add it as a *non-public* alias to avoid a
// name clash if the lib adds sha2 later for some other reason.
mod sha2_ {
    pub use sha2::*;
}

async fn router_with_fixture(fx: &Fixture) -> axum::Router {
    let state = test_state(fx.paths.clone());
    // Per-fixture log path so parallel tests don't collide on the
    // global default /tmp/rullama-page.log.
    let cfg = rullama_devserver::SecurityConfig {
        api_log_path: Some(fx._tempdir.path().join("api-log.txt")),
        ..Default::default()
    };
    build_app(state, cfg)
}

fn log_path_from_fixture(fx: &Fixture) -> std::path::PathBuf {
    fx._tempdir.path().join("api-log.txt")
}

fn req_get(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

fn req_head(uri: &str) -> Request<Body> {
    Request::builder()
        .method("HEAD")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn req_options(uri: &str) -> Request<Body> {
    Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn req_range(uri: &str, range: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .header(header::RANGE, range)
        .body(Body::empty())
        .unwrap()
}

async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    let collected = resp.into_body().collect().await.expect("collect body");
    collected.to_bytes().to_vec()
}

#[tokio::test]
async fn api_models_returns_synthetic_entry() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_get("/api/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get(header::CONTENT_TYPE).cloned();
    let body = body_bytes(resp).await;
    assert!(
        ct.map(|v| v.to_str().unwrap_or("").contains("application/json"))
            .unwrap_or(false),
        "content-type should be JSON"
    );
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let arr = value.as_array().expect("models is an array");
    assert!(
        !arr.is_empty(),
        "should discover at least the fixture model"
    );
    let m = &arr[0];
    assert_eq!(m["name"], "gemma4:e2b");
    assert_eq!(m["family"], "gemma4");
    assert_eq!(m["tag"], "e2b");
    assert_eq!(m["size"], fx.blob_size);
    assert_eq!(m["digest"], fx.digest);
    assert_eq!(m["filename"], format!("sha256-{}", fx.digest));
    assert_eq!(m["modelKey"], "gemma4:e2b");
    assert_eq!(m["multimodal"], false);
}

#[tokio::test]
async fn api_models_head_is_metadata_only() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_head("/api/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.contains("application/json"));
    let body = body_bytes(resp).await;
    assert!(body.is_empty(), "HEAD body must be empty");
}

#[tokio::test]
async fn api_blob_head_returns_metadata_headers() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_head("/api/blob/gemma4:e2b")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let headers = resp.headers();
    assert_eq!(
        headers.get("x-model-name").unwrap().to_str().unwrap(),
        "gemma4:e2b"
    );
    assert_eq!(
        headers.get("x-total-size").unwrap().to_str().unwrap(),
        fx.blob_size.to_string()
    );
    assert_eq!(headers.get("accept-ranges").unwrap(), "bytes");
    assert_eq!(
        headers.get("content-length").unwrap().to_str().unwrap(),
        fx.blob_size.to_string()
    );
    let body = body_bytes(resp).await;
    assert!(body.is_empty(), "HEAD body must be empty");
}

#[tokio::test]
async fn api_blob_full_get_streams_correct_bytes() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_get("/api/blob/gemma4:e2b")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(cl, fx.blob_size.to_string());
    let body = body_bytes(resp).await;
    assert_eq!(body.len() as u64, fx.blob_size);
    assert_eq!(&body[..GGUF_MAGIC.len()], GGUF_MAGIC);
}

#[tokio::test]
async fn api_blob_range_returns_206_with_content_range() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app
        .oneshot(req_range("/api/blob/gemma4:e2b", "bytes=0-15"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        resp.headers()
            .get(header::CONTENT_LENGTH)
            .unwrap()
            .to_str()
            .unwrap(),
        "16"
    );
    assert_eq!(
        resp.headers()
            .get(header::CONTENT_RANGE)
            .unwrap()
            .to_str()
            .unwrap(),
        format!("bytes 0-15/{}", fx.blob_size)
    );
    let body = body_bytes(resp).await;
    assert_eq!(body.len(), 16);
    assert_eq!(&body[..GGUF_MAGIC.len()], GGUF_MAGIC);
}

#[tokio::test]
async fn api_blob_range_open_ended_extends_to_eof() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let start = fx.blob_size - 4;
    let resp = app
        .oneshot(req_range(
            "/api/blob/gemma4:e2b",
            &format!("bytes={start}-"),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
    let cr = resp
        .headers()
        .get(header::CONTENT_RANGE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(
        cr,
        format!("bytes {}-{}/{}", start, fx.blob_size - 1, fx.blob_size)
    );
    let body = body_bytes(resp).await;
    assert_eq!(body.len(), 4);
}

#[tokio::test]
async fn api_blob_unknown_model_returns_404() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app
        .oneshot(req_get("/api/blob/does-not-exist:tag"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn options_preflight_returns_204() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app
        .oneshot(req_options("/api/blob/gemma4:e2b"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn post_api_log_appends_to_override_path() {
    let fx = build_fixture();
    let log_file = log_path_from_fixture(&fx);
    let app = router_with_fixture(&fx).await;
    let body = serde_json::json!({"tag":"smoke","msg":"hello"});
    let req = Request::builder()
        .method("POST")
        .uri("/api/log")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let contents = std::fs::read_to_string(&log_file).expect("log file written");
    assert!(
        contents.contains("[smoke] hello\n"),
        "expected `[smoke] hello\\n` in log, got: {contents:?}"
    );
}

#[tokio::test]
async fn pkg_serves_files_with_no_store_cache() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_get("/pkg/rullama.js")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let cc = resp
        .headers()
        .get(header::CACHE_CONTROL)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(cc.contains("no-store"));
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("javascript"));
    let body = body_bytes(resp).await;
    assert!(String::from_utf8_lossy(&body).contains("fixture"));
}

#[tokio::test]
async fn pkg_serves_wasm_with_correct_mime() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_get("/pkg/rullama_bg.wasm")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(ct, "application/wasm");
    let body = body_bytes(resp).await;
    assert!(body.starts_with(b"\x00asm"));
}

#[tokio::test]
async fn pkg_missing_returns_404() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app
        .oneshot(req_get("/pkg/does-not-exist.js"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ─── security-mode tests ──────────────────────────────────────────────

async fn router_public(fx: &Fixture) -> axum::Router {
    let state = test_state(fx.paths.clone());
    let cfg = rullama_devserver::SecurityConfig::public_defaults();
    build_app(state, cfg)
}

#[tokio::test]
async fn public_mode_disables_api_models() {
    let fx = build_fixture();
    let app = router_public(&fx).await;
    let resp = app.oneshot(req_get("/api/models")).await.unwrap();
    // With proxy disabled (serve_dist=true) AND no api/models route,
    // we fall through to the dist fallback. The fixture's dist dir is
    // empty / has no index.html, so the response is 500-or-404 — either
    // way, NOT 200 with a JSON model list.
    assert_ne!(resp.status(), StatusCode::OK, "should not expose models");
}

#[tokio::test]
async fn public_mode_rejects_api_log_writes() {
    let fx = build_fixture();
    let app = router_public(&fx).await;
    let body = serde_json::json!({"tag":"hostile","msg":"x"});
    let req = Request::builder()
        .method("POST")
        .uri("/api/log")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::NO_CONTENT,
        "should not accept log writes in public mode"
    );
}

#[tokio::test]
async fn coop_coep_corp_present_on_every_response() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let resp = app.oneshot(req_get("/api/models")).await.unwrap();
    let h = resp.headers().clone();
    assert_eq!(
        h.get("cross-origin-opener-policy")
            .unwrap()
            .to_str()
            .unwrap(),
        "same-origin"
    );
    assert_eq!(
        h.get("cross-origin-embedder-policy")
            .unwrap()
            .to_str()
            .unwrap(),
        "require-corp"
    );
    // /api/models gets cross-origin CORP so the page on a CF tunnel
    // hostname can fetch the model list from a localhost dev origin.
    assert_eq!(
        h.get("cross-origin-resource-policy")
            .unwrap()
            .to_str()
            .unwrap(),
        "cross-origin"
    );
}

#[tokio::test]
async fn cors_default_denies_unknown_origin() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    let req = Request::builder()
        .uri("/api/models")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert!(
        resp.headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none(),
        "no allow-origin should be echoed for unlisted origins"
    );
}

#[tokio::test]
async fn pkg_path_traversal_via_dotdot_components_blocked() {
    let fx = build_fixture();
    let app = router_with_fixture(&fx).await;
    // `../web/serve-tunnel.sh` would, before canonicalisation,
    // resolve to a real file. We expect 404.
    let resp = app
        .oneshot(req_get("/pkg/../web/serve-tunnel.sh"))
        .await
        .unwrap();
    assert!(
        resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::FORBIDDEN,
        "expected 404/403, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn api_log_body_size_limit_enforced() {
    let fx = build_fixture();
    let log_file = log_path_from_fixture(&fx);
    let app = router_with_fixture(&fx).await;
    // 16 KiB > the 8 KiB default cap → must be rejected before write.
    let oversized = "x".repeat(16 * 1024);
    let body = serde_json::json!({"tag":"big","msg":oversized});
    let req = Request::builder()
        .method("POST")
        .uri("/api/log")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert!(
        resp.status() == StatusCode::PAYLOAD_TOO_LARGE || resp.status() == StatusCode::BAD_REQUEST,
        "oversized POST should be rejected, got {}",
        resp.status()
    );
    // And nothing should have hit disk.
    let exists = std::fs::read_to_string(&log_file)
        .map(|c| !c.is_empty())
        .unwrap_or(false);
    assert!(!exists, "oversized payload should not have been written");
}
