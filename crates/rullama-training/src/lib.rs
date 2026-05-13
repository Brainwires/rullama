#![deny(missing_docs)]
//! `rullama-training` — placeholder for training-from-scratch primitives.
//!
//! No training-from-scratch code lives here yet. The crate exists as a
//! companion to `rullama-finetune` so the eventual split is already
//! reflected in the workspace:
//!
//! - **`rullama-finetune`** — local PEFT fine-tuning (LoRA / QLoRA / DoRA)
//!   on a pre-trained model, Burn-backed.
//! - **`rullama-training`** (this crate) — reserved for actual training
//!   from scratch (full-parameter pretraining, distributed training,
//!   eventual autodiff over rullama's own wgpu kernels). Add code here
//!   when that work begins.
//!
//! Vendored from `brainwires-framework`'s `brainwires-training` placeholder
//! during the move that turned rullama into a multi-crate Rust runtime.
