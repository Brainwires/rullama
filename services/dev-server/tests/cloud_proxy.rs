//! Cloud BYOK proxy route — hermetic checks driven via
//! `tower::ServiceExt::oneshot`. These cover the request-validation that
//! short-circuits BEFORE any upstream call (so no network, no key needed):
//! provider whitelist, the `X-Cloud-Key` requirement, and the `allow_cloud`
//! gate. Real upstream/SSE parity is verified out-of-band with curl
//! (proxy-vs-direct), per the plan.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tempfile::TempDir;
use tower::ServiceExt;

use rullama_devserver::{Paths, SecurityConfig, build_app, test_state};

/// Minimal repo stub `Paths::resolve` accepts. The cloud route never reads
/// the filesystem, but `test_state`/`Paths` still need a valid root.
fn build(allow_cloud: bool) -> (TempDir, axum::Router) {
    let td = TempDir::new().unwrap();
    let root = td.path().to_path_buf();
    std::fs::create_dir_all(root.join("crates")).unwrap();
    std::fs::create_dir_all(root.join("pkg")).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    let models = root.join("models");
    std::fs::create_dir_all(models.join("blobs")).unwrap();
    std::fs::create_dir_all(models.join("manifests")).unwrap();
    let paths = Paths::resolve(Some(root), Some(models)).unwrap();
    let cfg = SecurityConfig {
        allow_cloud,
        ..Default::default()
    };
    (td, build_app(test_state(paths), cfg))
}

fn post_chat(uri: &str, key: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(k) = key {
        b = b.header("x-cloud-key", k);
    }
    b.body(Body::from(r#"{"model":"x","messages":[]}"#))
        .unwrap()
}

#[tokio::test]
async fn unknown_provider_is_400() {
    let (_td, app) = build(true);
    // Provider is validated before any upstream call, so a key is present
    // but irrelevant — `foo` is not in the whitelist.
    let resp = app
        .oneshot(post_chat("/api/cloud/foo/chat", Some("sk-whatever")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn missing_key_is_401() {
    let (_td, app) = build(true);
    // Valid provider, no `X-Cloud-Key` → rejected before reaching the network.
    let resp = app
        .oneshot(post_chat("/api/cloud/openai/chat", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn empty_key_is_401() {
    let (_td, app) = build(true);
    let resp = app
        .oneshot(post_chat("/api/cloud/ollama/chat", Some("   ")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn disabled_cloud_route_is_absent() {
    // With `allow_cloud=false` the cloud router isn't mounted, so the request
    // falls through to the fallback. Use the `serve_dist` fallback (with a
    // stub index.html) so an unmatched extensionless path deterministically
    // yields the SPA shell (200). A *mounted* route would instead 401 on the
    // missing key — so 200 (and not 401) proves the route is absent.
    let td = TempDir::new().unwrap();
    let root = td.path().to_path_buf();
    std::fs::create_dir_all(root.join("crates")).unwrap();
    std::fs::create_dir_all(root.join("pkg")).unwrap();
    std::fs::create_dir_all(root.join("apps/web/dist")).unwrap();
    std::fs::write(root.join("apps/web/dist/index.html"), "<!doctype html>").unwrap();
    std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    let models = root.join("models");
    std::fs::create_dir_all(models.join("blobs")).unwrap();
    std::fs::create_dir_all(models.join("manifests")).unwrap();
    let paths = Paths::resolve(Some(root), Some(models)).unwrap();
    let cfg = SecurityConfig {
        allow_cloud: false,
        serve_dist: true,
        ..Default::default()
    };
    let app = build_app(test_state(paths), cfg);
    let resp = app
        .oneshot(post_chat("/api/cloud/openai/chat", None))
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "cloud route must be absent when allow_cloud=false"
    );
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn options_preflight_is_204() {
    let (_td, app) = build(true);
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/api/cloud/openai/chat")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
}
