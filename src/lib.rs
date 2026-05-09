//! rullama — Gemma 4 inference in the browser via wgpu + WASM.
//!
//! Crate root. The JS-facing surface lives in [`api`]; everything else is internal.

pub mod api;
pub mod backend;
pub mod error;
pub mod gguf;
pub mod kernels;
pub mod model;
pub mod multimodal;
pub mod reference;
pub mod sampling;
pub mod template;
pub mod tokenizer;

pub use error::RullamaError;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn __wasm_start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}
