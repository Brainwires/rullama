//! JS-facing types and entry points.
//!
//! On wasm32 these are exposed via wasm-bindgen. On native they remain Rust-only
//! and are used by integration tests.

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// M0 smoke export: doubles every f32 in `input` on the GPU and returns the result.
/// Lets JS confirm the WebGPU path is wired up before we ship real inference.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = computeSpike)]
pub async fn compute_spike_js(input: Vec<f32>) -> Result<Vec<f32>, JsError> {
    crate::backend::compute_spike(&input)
        .await
        .map_err(|e| JsError::new(&format!("{e}")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> u32 { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.95 }
fn default_top_k() -> u32 { 40 }
fn default_repetition_penalty() -> f32 { 1.0 }
