//! ROME Phase 1.4 — full edit pipeline CLI.
//!
//! Build a rank-1 adapter on `ffn_down` at a chosen layer that biases
//! the model toward producing the given target token when asked the
//! subject prompt. Writes safetensors bytes to
//! `RULLAMA_ROME_ADAPTER_PATH` (or `/tmp/rome.safetensors` by default).
//!
//! Usage:
//!
//! ```text
//! cargo run -p rullama --release --example rome_edit -- \
//!     ~/.ollama/models/blobs/sha256-<digest>           \
//!     5                                                \
//!     "What's the capital of France?"                  \
//!     "Brie"
//! ```
//!
//! Env knobs:
//!   - `RULLAMA_ROME_ALPHA` — edit step size (default 1.0). Bigger =
//!     stronger edit, but too big causes side effects on unrelated
//!     prompts. Layer-dependent — sweep this in Phase 1.5.
//!   - `RULLAMA_ROME_ADAPTER_PATH` — output path (default
//!     `/tmp/rome.safetensors`).
//!   - `RULLAMA_ROME_APPLY_CHAT_TEMPLATE=1` — wrap the subject prompt
//!     in `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`
//!     before encoding, mirroring the PWA chat path. Required for the
//!     edit to fire when loaded by the chat UI.
//!
//! After this completes:
//!   `cargo run -p rullama-finetune --release --example eval_adapter -- \
//!     <gguf> <adapter-path> "<prompt>"`
//! to see whether the edit fires.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use rullama::api::{ChatMessage, ChatRole, Model};

type BoxError = Box<dyn Error + Send + Sync>;

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: rome_edit <gguf-path> <layer> <subject-prompt> <target-text>".into()
        })?
        .into();
    let target_layer: u32 = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <layer>".into() })?
        .parse()?;
    let subject: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <subject-prompt>".into() })?;
    let target_text: String = args
        .next()
        .ok_or_else(|| -> BoxError { "missing <target-text>".into() })?;

    let alpha: f32 = env::var("RULLAMA_ROME_ALPHA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let out_path = env::var("RULLAMA_ROME_ADAPTER_PATH")
        .unwrap_or_else(|_| "/tmp/rome.safetensors".to_string());
    let apply_chat_template = env::var("RULLAMA_ROME_APPLY_CHAT_TEMPLATE").is_ok();

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    let subject_for_encoding = if apply_chat_template {
        let wrapped = model.render_chat_native(
            &[ChatMessage {
                role: ChatRole::User,
                content: subject.clone(),
            }],
            false,
        );
        eprintln!("[encode] chat-template wrapped subject:");
        eprintln!("        {wrapped:?}");
        wrapped
    } else {
        subject.clone()
    };
    let prompt_tokens = model.encode_tokens(&subject_for_encoding);
    eprintln!("[encode] subject = {} tokens", prompt_tokens.len());

    let target_tokens = model.encode_tokens(&target_text);
    if target_tokens.is_empty() {
        return Err("target_text tokenized to empty".into());
    }
    let target_token_id = target_tokens[0];
    let target_str = model.token_str_native(target_token_id).unwrap_or_default();
    eprintln!("[encode] target_token = {target_token_id} ({target_str:?})");

    eprintln!(
        "[rome] applying edit: layer={target_layer}, alpha={alpha}, target={target_str:?}…"
    );
    let safetensors_bytes = model
        .rome_edit_native(&prompt_tokens, target_layer, target_token_id, alpha)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    fs::write(&out_path, &safetensors_bytes)?;
    eprintln!(
        "[save] adapter → {} ({} bytes)",
        out_path,
        safetensors_bytes.len()
    );
    eprintln!();
    eprintln!("Now verify the edit fires:");
    eprintln!("  cargo run -p rullama-finetune --release --example eval_adapter -- \\");
    eprintln!("    {} {} \\", gguf_path.display(), out_path);
    eprintln!("    {:?}", subject);

    Ok(())
}
