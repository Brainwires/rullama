//! Reverse-proxy fallback to Vite for any non-/api/, non-/pkg/, non-/__rullama-dev-ws
//! request.
//!
//! Two paths:
//!   • HTTP — `client.request(req).await`, collect body, send back. Cheap,
//!     correct for the React app shell, modules, CSS, etc.
//!   • WebSocket upgrade — detect `Upgrade: websocket` on the inbound,
//!     open a raw TCP socket to Vite, replay the request line + headers,
//!     read the upstream's 101, return a matching 101 back through axum,
//!     then `copy_bidirectional` the upgraded inbound with the upstream
//!     TCP socket. This is what makes Vite HMR work through the
//!     devserver origin, so a code edit pushes live without having to
//!     load the page from `:5173` directly.
//!
//! Both paths only run when the devserver is in local-dev mode (the
//! `--public` switch removes this whole fallback and replaces it with
//! the `dist/` static serve).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, header, uri::PathAndQuery};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioIo;
use hyper_util::rt::TokioExecutor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::state::AppState;

pub async fn fallback_handler(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    if is_websocket_upgrade(req.headers()) {
        return proxy_ws_upgrade(state, req).await;
    }
    proxy_http(state, req).await
}

fn is_websocket_upgrade(headers: &HeaderMap) -> bool {
    let conn_has_upgrade = headers
        .get(header::CONNECTION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').any(|p| p.trim().eq_ignore_ascii_case("upgrade")))
        .unwrap_or(false);
    let upgrade_is_ws = headers
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    conn_has_upgrade && upgrade_is_ws
}

async fn proxy_http(state: Arc<AppState>, mut req: Request<Body>) -> Response {
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
            return (StatusCode::BAD_GATEWAY, format!("proxy: build uri failed: {e}"))
                .into_response();
        }
    };
    *req.uri_mut() = upstream;

    let h = req.headers_mut();
    for name in HOP_BY_HOP { h.remove(*name); }
    h.remove("host");

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

    let (parts, body) = resp.into_parts();
    let bytes = match body.collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("proxy: read upstream body: {e}")).into_response(),
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

/// WebSocket reverse-proxy. Connects upstream Vite over raw TCP, replays
/// the HTTP/1.1 upgrade handshake, then bridges the two sockets until
/// either side closes.
async fn proxy_ws_upgrade(state: Arc<AppState>, req: Request<Body>) -> Response {
    let port = state.vite_port;
    let path = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let method = req.method().clone();
    let headers = req.headers().clone();

    // Connect upstream.
    let mut upstream = match TcpStream::connect(("127.0.0.1", port)).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("ws-proxy: vite unreachable at 127.0.0.1:{port}: {e}"),
            )
                .into_response();
        }
    };
    // Disable Nagle — WebSocket frames are typically small and bursty;
    // delaying acks adds noticeable HMR latency.
    let _ = upstream.set_nodelay(true);

    // Serialize the inbound request as HTTP/1.1 onto the upstream socket.
    // We deliberately drop hop-by-hop headers but KEEP Connection +
    // Upgrade + the Sec-WebSocket-* family — those are the handshake.
    let mut wire = String::with_capacity(512);
    wire.push_str(method.as_str());
    wire.push(' ');
    wire.push_str(&path);
    wire.push_str(" HTTP/1.1\r\n");
    wire.push_str(&format!("Host: 127.0.0.1:{port}\r\n"));
    for (k, v) in headers.iter() {
        if k == header::HOST { continue; }
        if let Ok(s) = v.to_str() {
            wire.push_str(k.as_str());
            wire.push_str(": ");
            wire.push_str(s);
            wire.push_str("\r\n");
        }
    }
    wire.push_str("\r\n");
    if let Err(e) = upstream.write_all(wire.as_bytes()).await {
        return (StatusCode::BAD_GATEWAY, format!("ws-proxy: write handshake: {e}")).into_response();
    }

    // Read the upstream response (must be 101 Switching Protocols on the
    // happy path). We may over-read into the start of the WebSocket
    // stream — capture that tail and feed it to the inbound bridge
    // before starting `copy_bidirectional`, otherwise we'd lose the
    // upstream's first frame.
    let mut buf = vec![0u8; 8192];
    let mut total = 0usize;
    let (upstream_status, upstream_headers, prefix_tail) = loop {
        if total == buf.len() { buf.resize(buf.len() * 2, 0); }
        let n = match upstream.read(&mut buf[total..]).await {
            Ok(0) => {
                return (StatusCode::BAD_GATEWAY, "ws-proxy: vite closed during handshake")
                    .into_response();
            }
            Ok(n) => n,
            Err(e) => {
                return (StatusCode::BAD_GATEWAY, format!("ws-proxy: read handshake: {e}"))
                    .into_response();
            }
        };
        total += n;
        let mut hbuf = [httparse::EMPTY_HEADER; 64];
        let mut resp = httparse::Response::new(&mut hbuf);
        match resp.parse(&buf[..total]) {
            Ok(httparse::Status::Complete(consumed)) => {
                let status = resp.code.unwrap_or(502);
                let mut hm = HeaderMap::new();
                for h in resp.headers.iter() {
                    let Ok(name) = HeaderName::from_bytes(h.name.as_bytes()) else { continue };
                    let Ok(val) = HeaderValue::from_bytes(h.value) else { continue };
                    hm.append(name, val);
                }
                let tail = buf[consumed..total].to_vec();
                break (status, hm, tail);
            }
            Ok(httparse::Status::Partial) => continue,
            Err(e) => {
                return (StatusCode::BAD_GATEWAY, format!("ws-proxy: parse upstream response: {e}"))
                    .into_response();
            }
        }
    };

    // Build the response we return to the client. axum will then trigger
    // the inbound upgrade so we can take over the socket.
    let status = match StatusCode::from_u16(upstream_status) {
        Ok(s) => s,
        Err(_) => StatusCode::BAD_GATEWAY,
    };
    let mut builder = Response::builder().status(status);
    for (k, v) in upstream_headers.iter() {
        // Skip Content-Length (would be wrong) and Transfer-Encoding
        // (irrelevant after upgrade). Keep Upgrade + Connection — those
        // are what tell the browser the upgrade completed.
        if matches!(k.as_str(), "content-length" | "transfer-encoding") { continue; }
        builder = builder.header(k, v);
    }
    let our_resp = match builder.body(Body::empty()) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("ws-proxy: build response: {e}"),
            )
                .into_response();
        }
    };

    // If upstream didn't upgrade, don't try to bridge — just hand the
    // status back as-is. (E.g. Vite returned 400 because the
    // Sec-WebSocket-Key was malformed.)
    if status != StatusCode::SWITCHING_PROTOCOLS {
        return our_resp;
    }

    // Schedule the bridge. `hyper::upgrade::on(req)` resolves once axum
    // has finished sending our 101 to the inbound and the socket is ours.
    let on_upgrade = hyper::upgrade::on(req);
    tokio::spawn(async move {
        let inbound = match on_upgrade.await {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!("ws-proxy: inbound upgrade failed: {e}");
                return;
            }
        };
        let mut inbound = TokioIo::new(inbound);
        // Push the prefix bytes we over-read while parsing the 101.
        if !prefix_tail.is_empty() {
            if let Err(e) = inbound.write_all(&prefix_tail).await {
                tracing::warn!("ws-proxy: write prefix: {e}");
                return;
            }
        }
        // Bidirectional copy until either side closes.
        match tokio::io::copy_bidirectional(&mut inbound, &mut upstream).await {
            Ok((tx, rx)) => tracing::debug!("ws-proxy: closed (client→up {tx} B, up→client {rx} B)"),
            Err(e) => tracing::debug!("ws-proxy: closed with error: {e}"),
        }
    });
    our_resp
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
