//! Multimodal towers for Gemma 4 (vision + audio).
//!
//! Mirrors Ollama's `model/models/gemma4/{model_vision,model_audio,process_image,process_audio}.go`.
//! The text language model is unchanged — multimodal features become "soft tokens"
//! injected into the residual stream via `Forward::step_with_embedding`.

pub mod vision;

pub use vision::{VisionConfig, VisionForward};
