//! /__rullama-dev-ws — WebSocket fan-out for `DevEvent`s.
//!
//! The browser-side client (`examples/web/src/lib/dev-hmr.ts`) subscribes;
//! the watcher publishes. Each connected tab gets every event from the
//! moment it connects. No replay of history.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};

use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/__rullama-dev-ws", get(upgrade))
}

async fn upgrade(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle(socket, state))
}

async fn handle(socket: WebSocket, state: Arc<AppState>) {
    let (mut tx, mut rx) = socket.split();
    let mut events = state.events.subscribe();
    tracing::info!("[ws] client connected");

    // Send a hello so the browser knows the WS is up (helpful for the
    // dev-hmr.ts client to render its "connected" indicator).
    if tx.send(Message::Text(
        serde_json::json!({"type":"hello","at_ms":now_ms()}).to_string()
    )).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            // Server-side push.
            ev = events.recv() => {
                match ev {
                    Ok(ev) => {
                        let s = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".to_string());
                        if tx.send(Message::Text(s)).await.is_err() { break; }
                    }
                    Err(_lag) => { /* lagged or closed; ignore */ }
                }
            }
            // Client → server: just consume to keep the connection alive;
            // we don't act on anything the client says.
            msg = rx.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
    tracing::info!("[ws] client disconnected");
}

fn now_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}
