//! JS-facing types and entry points.
//!
//! On wasm32 these are exposed via wasm-bindgen. On native they remain Rust-only
//! and are used by integration tests / examples.
//!
//! Minimal API surface (M5 v0):
//!   - `Model::load(bytes)` — parse GGUF, init wgpu, upload pipelines (no weights yet).
//!   - `Model::encode(text)` / `Model::token_str(id)` — tokenizer access.
//!   - `Model::step(token_id)` — feed a single token at the current position; returns
//!     the argmax of the resulting next-token logits. Mutates internal KV cache.
//!   - `Model::reset()` — clear KV state to start a fresh conversation.
//!   - `Model::is_eos(id)` — checks against the GGUF's eos token id list.
//!
//! Streaming is JS's responsibility: loop `step` and call `token_str(id)` per step.
//! A `ReadableStream<string>` wrapper lands in v0.2.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::GgufReader;
use crate::model::config::Gemma4Config;
use crate::multimodal::{AudioConfig, GpuAudioForward, VisionConfig, VisionForward, decode_wav};
use crate::reference::Weights;
use crate::reference::forward_chained::Forward;
use crate::sampling::{Sampler, SamplingOptions};
use crate::template::gemma4_small;
use crate::tokenizer::BpeTokenizer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// M0 smoke export: doubles every f32 on the GPU. Useful from JS to confirm WebGPU
/// is wired up before loading the full model.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = computeSpike)]
pub async fn compute_spike_js(input: Vec<f32>) -> std::result::Result<Vec<f32>, JsError> {
    crate::backend::compute_spike(&input)
        .await
        .map_err(|e| JsError::new(&format!("{e}")))
}

/// Hyperparameters for [`Model::rome_edit_iterative_native`]. Defaults
/// mirror EasyEdit's Llama-3.2-3B config (the closest scale analog to
/// Gemma 4 e2b) — see
/// `https://github.com/zjunlp/EasyEdit/blob/main/hparams/ROME/llama3.2-3b.yaml`.
#[derive(Clone, Copy, Debug)]
pub struct RomeIterativeHparams {
    pub num_steps: u32,
    pub v_lr: f32,
    pub v_weight_decay: f32,
    pub clamp_norm_factor: f32,
    pub kl_factor: f32,
    pub early_stop: f32,
}

impl Default for RomeIterativeHparams {
    fn default() -> Self {
        Self {
            num_steps: 25,
            v_lr: 0.5,
            v_weight_decay: 1e-3,
            clamp_norm_factor: 4.0,
            kl_factor: 0.0625,
            early_stop: 5e-2,
        }
    }
}

// ---------- public Model surface ----------

/// A loaded Gemma 4 model with all GPU resources allocated. One `Model` corresponds to
/// one conversation: it owns the KV cache and tracks the current position.
///
/// Internally a `Model` is a tokenizer + a [`Forward`] + a [`Sampler`]. `Forward` runs
/// one wgpu CommandEncoder per token (M7 work) — significantly faster than the original
/// per-kernel-readback path, which is now retained only as a parity oracle.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Model {
    tokenizer: BpeTokenizer,
    forward: Forward,
    /// Vision tower — lazily allocated. `None` either because the GGUF
    /// has no vision tensors *or* because `release_vision_weights`
    /// dropped the previous instance to free its ~250 MB scratch.
    /// `vision_capable` distinguishes the two: a `None + capable=true`
    /// state is the "released, will be rebuilt on next encode" case.
    vision: Option<VisionForward>,
    /// True iff the loaded GGUF carries the vision tensors (presence
    /// of `v.patch_embd.weight`). Stable for the lifetime of the
    /// Model — `hasVision` reports this, not `vision.is_some()`, so
    /// releasing the tower doesn't make the UI think vision is
    /// unavailable.
    vision_capable: bool,
    /// Audio tower — same lazy-allocation contract as `vision`.
    audio: Option<GpuAudioForward>,
    /// True iff the loaded GGUF carries the audio tensors.
    audio_capable: bool,
    sampler: Sampler,
    /// Cooperative cancel flag for in-flight multimodal encodes. Flipped
    /// via `cancelMultimodalEncode()` from JS; the vision and audio
    /// encoders check this between transformer layers and bail with
    /// `RullamaError::Cancelled`. Cleared at the start of each encode so
    /// a stale flag from a previous cancel doesn't poison the next call.
    encode_cancel: Arc<AtomicBool>,
    /// Active LoRA adapter, if any. Set via `load_adapter_native` /
    /// `loadAdapter`; cleared via `clear_adapter_native` /
    /// `clearAdapter`. When `Some`, `step_native` routes through
    /// `Forward::step_with_lora` so chat output reflects the adapter.
    adapter: Option<crate::lora::InferenceAdapter>,
}

impl Model {
    /// Build a Model from an already-constructed GGUF reader. Shared by both
    /// the in-memory and streaming entry points so they can't drift.
    async fn from_reader(reader: GgufReader) -> Result<Self> {
        Self::from_reader_with_modes(
            reader,
            true,
            true,
            crate::reference::forward_chained::MAX_CONTEXT,
        )
        .await
    }

    /// Like [`from_reader`] but lets the caller skip the vision and/or audio
    /// tower construction and cap the KV-cache pre-allocation. Useful on
    /// memory-constrained targets (e.g. iPhone 16e shared 8 GB RAM) where
    /// eagerly building `VisionForward` / `GpuAudioForward` + a 4096-token
    /// KV cache would push the WebContent process over Jetsam and the page
    /// crashes during wasm-load or the first inference step.
    async fn from_reader_with_modes(
        reader: GgufReader,
        with_vision: bool,
        with_audio: bool,
        max_context: u32,
    ) -> Result<Self> {
        let cfg = Gemma4Config::from_gguf(&reader)?;
        let tokenizer = BpeTokenizer::from_gguf(&reader)?;
        let d_text = cfg.d_model;
        let r_arc = Arc::new(reader);
        let weights = Weights::new(r_arc.clone());
        let ctx = WgpuCtx::new().await?;
        let pipes = Arc::new(Pipelines::new_with_features(
            &ctx.device,
            ctx.has_subgroups,
            ctx.has_f16,
        ));
        let wcache = Arc::new(WeightCache::new(
            r_arc.clone(),
            ctx.device.clone(),
            ctx.queue.clone(),
        ));

        // Detect vision tower (presence of v.patch_embd.weight). Build VisionForward
        // before consuming `ctx`/`pipes`/`wcache` into the text Forward.
        let vision_capable = r_arc.tensor("v.patch_embd.weight").is_ok();
        let vision = if with_vision && vision_capable {
            let vcfg = VisionConfig::from_gguf(&r_arc, d_text)?;
            Some(VisionForward::new(vcfg, ctx.clone(), pipes.clone(), wcache.clone()).await?)
        } else {
            None
        };

        // Detect audio tower (presence of a.conv1d.0.weight). The GPU
        // encoder runs the 12 Conformer blocks + projector on the GPU; mel
        // features + SSCP convs + pre-encode linear stay on CPU (small, and
        // their data layouts don't pay off vs the bulk of the work).
        let audio_capable = r_arc.tensor("a.conv1d.0.weight").is_ok();
        let audio = if with_audio && audio_capable {
            let acfg = AudioConfig::from_gguf(&r_arc, d_text)?;
            Some(GpuAudioForward::new(acfg, ctx.clone(), pipes.clone(), wcache.clone()).await?)
        } else {
            None
        };

        let forward =
            Forward::new_with_max_context(cfg, ctx, pipes, weights, wcache, max_context).await?;
        Ok(Self {
            tokenizer,
            forward,
            vision,
            vision_capable: with_vision && vision_capable,
            audio,
            audio_capable: with_audio && audio_capable,
            sampler: Sampler::new(SamplingOptions::default()),
            encode_cancel: Arc::new(AtomicBool::new(false)),
            adapter: None,
        })
    }

    /// True iff this checkpoint carries a vision tower (gemma4:e2b/e4b).
    /// Stable for the lifetime of the Model — returns `true` even when
    /// `release_vision_weights` has temporarily dropped the tower
    /// (the next encode will rebuild it).
    pub fn has_vision_native(&self) -> bool {
        self.vision_capable
    }

    /// Ensure the vision tower is allocated. Re-builds the
    /// `VisionForward` struct (allocating ~250 MB of per-image
    /// scratch buffers) if a prior `release_vision_weights` dropped
    /// it. No-op when the tower is already live or the GGUF has no
    /// vision tensors.
    async fn ensure_vision(&mut self) -> Result<()> {
        if self.vision.is_some() || !self.vision_capable {
            return Ok(());
        }
        let reader = self.forward.wcache().reader_arc();
        let d_text = self.forward.cfg().d_model;
        let ctx = self.forward.ctx().clone();
        let pipes = self.forward.pipes().clone();
        let wcache = self.forward.wcache().clone();
        let vcfg = VisionConfig::from_gguf(&reader, d_text)?;
        self.vision = Some(VisionForward::new(vcfg, ctx, pipes, wcache).await?);
        Ok(())
    }

    /// Encode an RGB image into a flat sequence of soft-token embeddings.
    ///
    /// `pixels`: `[3 * h * w]` f32, channel-first `[R..., G..., B...]`, normalised
    /// to `[-1, 1]`. `h` and `w` must be multiples of `patch_size * n_merge` (= 48).
    /// Returns `[n_pooled_patches * d_text]` f32 — one row of d_text per soft token.
    ///
    /// Rebuilds the vision tower if a prior `release_vision_weights`
    /// dropped it — see `ensure_vision`. This is `&mut self` (was
    /// `&self`) so the rebuild can happen without interior mutability.
    pub async fn encode_image_native(
        &mut self,
        pixels: &[f32],
        h: usize,
        w: usize,
        progress: Option<&dyn Fn(u32, u32)>,
    ) -> Result<Vec<f32>> {
        self.ensure_vision().await?;
        let v = self.vision.as_ref().ok_or_else(|| {
            crate::error::RullamaError::Inference(
                "encode_image: this checkpoint has no vision tower".into(),
            )
        })?;
        // Clear any flag left over from a previous cancel so it doesn't
        // poison this encode.
        self.encode_cancel.store(false, Ordering::Relaxed);
        v.encode(pixels, h, w, progress, Some(self.encode_cancel.clone()))
            .await
    }

    /// Number of soft tokens an image of `h × w` pixels produces (after AvgPool 3×3
    /// of patch grid). Useful for sizing prompt buffers without running the encoder.
    ///
    /// Falls back to deriving from `patch_size=16`, `n_merge=3` (the
    /// gemma4 vision constants) when the tower has been released and
    /// the cfg isn't reachable through a `VisionForward` instance.
    pub fn image_soft_token_count_native(&self, h: usize, w: usize) -> Option<usize> {
        if !self.vision_capable {
            return None;
        }
        let align: usize = match self.vision.as_ref() {
            Some(v) => {
                let cfg = v.cfg();
                (cfg.patch_size * cfg.n_merge) as usize
            }
            None => 48, // gemma4 constants: patch_size=16, n_merge=3
        };
        if !h.is_multiple_of(align) || !w.is_multiple_of(align) {
            return None;
        }
        let pooled_h = h / align;
        let pooled_w = w / align;
        Some(pooled_h * pooled_w)
    }

    /// True iff this checkpoint carries an audio tower. Like
    /// `has_vision_native`, stable across `release_audio_weights`.
    pub fn has_audio_native(&self) -> bool {
        self.audio_capable
    }

    /// Re-build the `GpuAudioForward` struct if `release_audio_weights`
    /// dropped it. Mirrors [`Self::ensure_vision`].
    async fn ensure_audio(&mut self) -> Result<()> {
        if self.audio.is_some() || !self.audio_capable {
            return Ok(());
        }
        let reader = self.forward.wcache().reader_arc();
        let d_text = self.forward.cfg().d_model;
        let ctx = self.forward.ctx().clone();
        let pipes = self.forward.pipes().clone();
        let wcache = self.forward.wcache().clone();
        let acfg = AudioConfig::from_gguf(&reader, d_text)?;
        self.audio = Some(GpuAudioForward::new(acfg, ctx, pipes, wcache).await?);
        Ok(())
    }

    /// Encode raw 16 kHz mono PCM (`Vec<f32>` in `[-1, 1]`) into a flat sequence
    /// of soft-token embeddings. Returns `[n_audio_tokens * d_text]` f32.
    pub async fn encode_audio_native(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.ensure_audio().await?;
        let a = self.audio.as_ref().ok_or_else(|| {
            crate::error::RullamaError::Inference(
                "encode_audio: this checkpoint has no audio tower".into(),
            )
        })?;
        self.encode_cancel.store(false, Ordering::Relaxed);
        a.encode(pcm, Some(self.encode_cancel.clone())).await
    }

    /// Flip the cooperative cancel flag for any in-flight multimodal
    /// encode. The vision and audio loops check this between layer
    /// dispatches and bail with `RullamaError::Cancelled`. No-op when
    /// no encode is running; the flag is cleared at the start of the
    /// next encode either way.
    pub fn cancel_multimodal_encode_native(&self) {
        self.encode_cancel.store(true, Ordering::Relaxed);
    }

    /// Decode a WAV file (RIFF/WAVE PCM 8/16/24/32 or float32) into 16 kHz
    /// mono `Vec<f32>`. Helper for callers that want to feed `encode_audio`.
    pub fn decode_wav_native(bytes: &[u8]) -> Result<Vec<f32>> {
        decode_wav(bytes)
    }

    /// `(begin_id, end_id)` for the `<|audio>` / `<audio|>` sentinels if both
    /// exist in the tokenizer vocab; else `None`. Native equivalent of the JS
    /// `audioSentinelIds` shim.
    pub fn audio_sentinel_ids_native(&self) -> Option<(u32, u32)> {
        let begin = self.tokenizer.str_to_id("<|audio>")?;
        let end = self.tokenizer.str_to_id("<audio|>")?;
        Some((begin, end))
    }

    /// `(begin_id, end_id)` for the `<|image>` / `<image|>` sentinels.
    pub fn image_sentinel_ids_native(&self) -> Option<(u32, u32)> {
        let begin = self.tokenizer.str_to_id("<|image>")?;
        let end = self.tokenizer.str_to_id("<image|>")?;
        Some((begin, end))
    }

    /// Evict the vision tower entirely — both the cached weights
    /// (~650 MB on gemma4:e2b) AND the `VisionForward` struct's
    /// per-image scratch (~250 MB of `MAX_PATCHES`-sized intermediates
    /// that `drop_prefix` alone won't touch because they're owned
    /// fields on the struct, not entries in `WeightCache`). Returns
    /// the number of cache entries freed.
    ///
    /// `hasVision` keeps returning `true` after this call — the next
    /// `encode_image` rebuilds the tower automatically via
    /// `ensure_vision`. The rebuild allocates the scratch buffers but
    /// doesn't upload weights until the encode itself touches them
    /// (lazy `WeightCache::buffer_async` path).
    ///
    /// Used on memory-constrained devices (iPhone Safari WebContent
    /// ~3 GB cap) where holding text weights + vision scratch +
    /// vision weights + KV cache simultaneously exceeds the budget.
    pub fn release_vision_weights_native(&mut self) -> usize {
        let freed = {
            let wc = self.forward.wcache();
            wc.drop_prefix("v.") + wc.drop_prefix("mm.input_projection")
        };
        // Dropping `vision` releases the `MAX_PATCHES`-sized
        // intermediates (~250 MB) that `drop_prefix` can't reach.
        self.vision = None;
        freed
    }

    /// Symmetric to [`release_vision_weights_native`]: drops cached
    /// audio-tower weights AND the `GpuAudioForward` struct's scratch.
    pub fn release_audio_weights_native(&mut self) -> usize {
        let freed = {
            let wc = self.forward.wcache();
            wc.drop_prefix("a.") + wc.drop_prefix("mm.a.")
        };
        self.audio = None;
        freed
    }

    /// Re-allocate the per-layer KV cache at a smaller (or larger) capacity.
    /// Returns the *previous* `max_context` so the caller can restore on
    /// demand. Discards any cached KV content (kv_lens reset to 0, pos = 0).
    ///
    /// Use case: chat reserves `max_context` positions (~600 MB at 4096 on
    /// gemma4:e2b) which training's NextToken loss only needs 1 position
    /// of. The browser-side `trainingStart` handler calls this before
    /// `TrainingSession::new` to hand the freed memory to training scratch;
    /// `trainingFinish` calls it again with the saved original value to
    /// restore chat's full cache.
    pub fn shrink_kv_native(&mut self, new_max_context: u32) -> Result<u32> {
        self.forward.shrink_kv(new_max_context)
    }

    /// Current per-layer KV cache capacity (in tokens). Snapshot before
    /// `shrink_kv_native` so you know what to restore.
    pub fn max_context_native(&self) -> u32 {
        self.forward.max_context()
    }

    /// Total bytes currently held in the shared `WeightCache`. Useful for
    /// memory accounting / regression checks around `release_*_weights`.
    pub fn cached_weight_bytes_native(&self) -> u64 {
        self.forward.wcache().cached_bytes()
    }

    /// Native-friendly constructor: takes ownership of GGUF bytes, initializes WebGPU,
    /// and prepares all the on-GPU resources (compute pipelines, weight cache).
    pub async fn load_native(bytes: Vec<u8>) -> Result<Self> {
        let reader = GgufReader::new(bytes)?;
        Self::from_reader(reader).await
    }

    /// Streaming constructor: takes any [`crate::gguf::TensorFetcher`] (in-memory or
    /// HTTP) and reads only the header up front. Tensor bytes are pulled lazily
    /// through the fetcher and dropped after each GPU upload — this is what keeps
    /// peak CPU memory bounded for the wasm32 4 GB linear-memory cap.
    pub async fn load_streaming(
        fetcher: std::sync::Arc<dyn crate::gguf::TensorFetcher>,
    ) -> Result<Self> {
        let reader = GgufReader::new_streaming(fetcher).await?;
        Self::from_reader(reader).await
    }

    /// Streaming load with an explicit KV-cache cap but vision + audio
    /// towers still built (when the GGUF carries them). Lets a mobile
    /// caller load a multimodal model with a smaller KV pre-alloc —
    /// e.g. iPhone passes `max_context = 2048` and saves ~600 MB
    /// against the compile-time `MAX_CONTEXT = 4096` budget. `0` keeps
    /// the default.
    pub async fn load_streaming_with_max_context(
        fetcher: std::sync::Arc<dyn crate::gguf::TensorFetcher>,
        max_context: u32,
    ) -> Result<Self> {
        let reader = GgufReader::new_streaming(fetcher).await?;
        let cap = if max_context == 0 {
            crate::reference::forward_chained::MAX_CONTEXT
        } else {
            max_context
        };
        Self::from_reader_with_modes(reader, true, true, cap).await
    }

    /// Text-only streaming load. Skips the vision and audio towers even if the
    /// GGUF contains them and caps the KV cache to `max_context` tokens
    /// (rather than the compile-time `MAX_CONTEXT = 4096`). The pair makes
    /// the difference between "iPhone Safari WebContent process gets killed
    /// mid-load" and "model loads and generates tokens." 512 is a fine
    /// default for chat-bot-sized turns on a phone.
    pub async fn load_streaming_text_only(
        fetcher: std::sync::Arc<dyn crate::gguf::TensorFetcher>,
        max_context: u32,
    ) -> Result<Self> {
        let reader = GgufReader::new_streaming(fetcher).await?;
        Self::from_reader_with_modes(reader, false, false, max_context).await
    }

    /// Encode text → token IDs (Ollama-matching BPE).
    pub fn encode_tokens(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Look up a token ID's string form (raw vocab entry; SentencePiece `▁` markers
    /// are not stripped — the caller does that in JS if it wants display text).
    pub fn token_str_native(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_str(id).map(|s| s.to_string())
    }

    /// ROME helper: locate the index of the LAST token belonging to
    /// `subject` within `prompt_tokens`. Mirrors EasyEdit's
    /// `find_fact_lookup_idx` with `fact_token = "subject_last"`.
    ///
    /// Strategy: walk forward through the prompt, accumulating decoded
    /// text. After each token, check whether the accumulated text
    /// ends with the subject (ignoring SentencePiece `▁` markers and
    /// case). The LAST such match is the subject-last position —
    /// matches ROME's "if the subject appears more than once, edit at
    /// the most recent mention" behavior.
    ///
    /// Returns `None` if the subject is not found in `prompt_tokens`'s
    /// decoded form.
    pub fn find_subject_last_pos(&self, prompt_tokens: &[u32], subject: &str) -> Option<u32> {
        let subject_norm = subject.trim().to_lowercase();
        if subject_norm.is_empty() {
            return None;
        }
        let mut acc = String::new();
        let mut best: Option<usize> = None;
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            if let Some(s) = self.tokenizer.id_to_str(tok) {
                // SentencePiece-style space prefix → real space; some
                // Gemma tokens also embed control chars we ignore.
                let s = s.replace('▁', " ");
                acc.push_str(&s);
            }
            // Normalize for matching: collapse whitespace, lowercase,
            // strip trailing whitespace so subject "France" matches
            // both "...of France" and "...of France?" (the "?" is in
            // a later token).
            let acc_norm: String = acc
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            if acc_norm.ends_with(&subject_norm) {
                best = Some(i);
            }
        }
        best.map(|i| i as u32)
    }

    /// Number of tokens in the vocab.
    pub fn vocab_size_native(&self) -> u32 {
        self.forward.cfg().vocab_size
    }

    /// Current sequence position (number of tokens fed so far).
    pub fn position_native(&self) -> u32 {
        self.forward.pos()
    }

    /// True iff `id` is one of the GGUF's EOS / EOT / end-of-turn tokens.
    pub fn is_eos_native(&self, id: u32) -> bool {
        self.forward.cfg().eos_ids.contains(&id)
    }

    /// Reset KV state so the next call starts from an empty conversation.
    /// Mutable handle on the underlying text `Forward`. Exposed for the
    /// training crate (`rullama-finetune::TrainingSession`) so it can
    /// drive `step_capture` and `backward_step` on the same model the
    /// inference path uses.
    pub fn forward_mut(&mut self) -> &mut Forward {
        &mut self.forward
    }
    /// Immutable handle on the text `Forward`.
    pub fn forward(&self) -> &Forward {
        &self.forward
    }

    pub fn reset_native(&mut self) {
        self.forward.reset();
        self.sampler.clear_history();
    }

    /// Snapshot KV cache + position + sampler state into a single byte
    /// blob suitable for OPFS-backed suspend/resume. Layout:
    ///
    /// ```text
    ///   [0..4]   magic = "RLMS"
    ///   [4]      version = 1
    ///   [5..8]   reserved
    ///   [8..12]  sampler_len (u32 LE)
    ///   [12..16] kv_len (u32 LE)
    ///   [16..16+sampler_len]      sampler bytes (Sampler::dump_state)
    ///   [16+sampler_len..]        kv bytes      (Forward::dump_kv)
    /// ```
    ///
    /// On resume both pieces must be applied together — the sampler RNG
    /// state matters for non-greedy sampling determinism (matching the
    /// trajectory the user was already seeing).
    pub async fn save_kv_state_native(&self) -> Result<Vec<u8>> {
        let sampler_bytes = self.sampler.dump_state();
        let kv_bytes = self.forward.dump_kv().await?;
        let mut out = Vec::with_capacity(16 + sampler_bytes.len() + kv_bytes.len());
        out.extend_from_slice(b"RLMS");
        out.push(1u8);
        out.extend_from_slice(&[0u8, 0u8, 0u8]);
        out.extend_from_slice(&(sampler_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(kv_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&sampler_bytes);
        out.extend_from_slice(&kv_bytes);
        Ok(out)
    }

    /// Inverse of [`save_kv_state_native`]. Applies sampler state first
    /// (cheap), then KV state (writes 26 layers × 2 buffers to GPU). On
    /// any validation error the model state is left untouched and the
    /// caller can fall back to token replay.
    pub fn restore_kv_state_native(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 16 || &bytes[0..4] != b"RLMS" {
            return Err(crate::error::RullamaError::Inference(
                "model state snapshot: bad magic".into(),
            ));
        }
        let version = bytes[4];
        if version != 1 {
            return Err(crate::error::RullamaError::Inference(format!(
                "model state snapshot: unknown version {version}"
            )));
        }
        let sampler_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let kv_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let sampler_off = 16usize;
        let kv_off = sampler_off + sampler_len;
        if bytes.len() < kv_off + kv_len {
            return Err(crate::error::RullamaError::Inference(format!(
                "model state snapshot: truncated (have {}, need {})",
                bytes.len(),
                kv_off + kv_len,
            )));
        }
        // Validate KV first (it's the larger / more failure-prone piece);
        // we can do this without mutating state because load_kv only
        // mutates after it has validated.
        self.forward.load_kv(&bytes[kv_off..kv_off + kv_len])?;
        self.sampler
            .load_state(&bytes[sampler_off..sampler_off + sampler_len])
            .map_err(|e| crate::error::RullamaError::Inference(format!("sampler restore: {e}")))?;
        Ok(())
    }

    /// Configure sampling. Defaults: temperature=0.7, top_k=40, top_p=0.95, no rep penalty.
    pub fn set_sampling_native(&mut self, opts: SamplingOptions) {
        self.sampler.set_options(opts);
    }

    /// Feed one token at the current position. Returns the *sampled* next token id
    /// (using current SamplingOptions). With `temperature=0`, this is the argmax.
    ///
    /// Routes through `Forward::step_with_lora` automatically when an
    /// inference adapter is active (see [`Self::load_adapter_native`]).
    pub async fn step_native(&mut self, token_id: u32) -> Result<u32> {
        self.sampler.observe(token_id);
        let logits = match &self.adapter {
            Some(adapter) => {
                let slots = adapter.layer_slots(self.forward.cfg().n_layers);
                self.forward.step_with_lora(token_id, &slots).await?
            }
            None => self.forward.step(token_id).await?,
        };
        let next = self.sampler.sample(&logits);
        Ok(next)
    }

    /// True iff a LoRA adapter is currently active. Browser chat code
    /// uses this to surface the "with adapter" badge.
    pub fn has_adapter_native(&self) -> bool {
        self.adapter.is_some()
    }

    /// Number of LoRA slots in the active adapter (zero if none).
    pub fn adapter_slot_count_native(&self) -> usize {
        self.adapter.as_ref().map(|a| a.len()).unwrap_or(0)
    }

    /// Load a safetensors-formatted LoRA adapter from a byte buffer and
    /// make it active. Replaces any previously-loaded adapter.
    ///
    /// The adapter must have been produced by
    /// `TrainingSession::save_adapter_to_bytes` (or compatible) — the
    /// loader reads the metadata sidecar's `rank` / `alpha` /
    /// `target_modules` and allocates GPU buffers sized against this
    /// model's config. Mismatched dims surface a `RullamaError::Inference`.
    pub fn load_adapter_native(&mut self, bytes: &[u8]) -> Result<usize> {
        let ctx = Arc::new(self.forward.ctx().clone());
        let cfg = self.forward.cfg().clone();
        let adapter = crate::lora::InferenceAdapter::from_safetensors_bytes(ctx, &cfg, bytes)?;
        let n = adapter.len();
        self.adapter = Some(adapter);
        Ok(n)
    }

    /// Drop the active adapter (subsequent generation uses base weights only).
    pub fn clear_adapter_native(&mut self) {
        self.adapter = None;
    }

    /// **ROME Phase 1.1 — `k*` extraction.**
    ///
    /// Run the prompt through the model, capture the post-GEGLU
    /// activation (`ffn_act`) at `target_layer` for the LAST prompt
    /// token, and return it as a `[d_ffn]` f32 vector. This is
    /// exactly `k*` from the ROME paper (Meng et al. 2022) — the
    /// "subject's key" vector that addresses fact storage in the
    /// FFN's down-projection. It's specifically the INPUT to
    /// `ffn_down`, so its shape matches the rank-1 update factor
    /// `A` when `ffn_down` is the edited matrix.
    ///
    /// After this call, `target_layer` and `k*` are the inputs to
    /// `compute_rome_v_star` (Phase 1.2) which finds the target
    /// `ffn_down` output that flips the model's prediction.
    pub async fn extract_mlp_input_native(
        &mut self,
        prompt_tokens: &[u32],
        target_layer: u32,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            return Err(crate::error::RullamaError::Inference(
                "extract_mlp_input_native: prompt_tokens must be non-empty".into(),
            ));
        }
        let n_layers = self.forward.cfg().n_layers;
        if target_layer >= n_layers {
            return Err(crate::error::RullamaError::Inference(format!(
                "extract_mlp_input_native: target_layer {target_layer} out of range (have {n_layers})"
            )));
        }
        let ctx = Arc::new(self.forward.ctx().clone());
        let cfg = self.forward.cfg().clone();
        // Size capture buffers for the prompt length. `step_capture`
        // writes activations at per-position offsets, so the buffers
        // must be seq-shaped to hold the last token's slice.
        let seq_len = prompt_tokens.len() as u32;
        let capture = crate::reference::rome::RomeCapture::new(&ctx, &cfg, seq_len);

        // Reset KV so the prompt runs from position 0.
        self.forward.reset();

        // Forward EVERY prompt token via `step_capture`. The kernels
        // index captures by current KV position, so unless every step
        // writes into our buffers the final position's slice will be
        // wrong (we only care about the LAST position's, but the
        // forward path indexes by `pos()` which advances every step).
        let captures = capture.as_captures();
        for &tok in prompt_tokens {
            let _ = self
                .forward
                .step_capture(tok, &captures, None)
                .await?;
        }
        let last_position = (prompt_tokens.len() - 1) as u32;
        drop(captures);

        capture.read_ffn_act(target_layer, last_position).await
    }

    /// **ROME Phase 1.2 — first-order v\* gradient.**
    ///
    /// Compute `∂loss/∂ffn_out[target_layer]` at the subject prompt's
    /// last token, where `loss = -log P(target_token | prompt)`. This
    /// is the direction in `[d_model]` space we should move
    /// `ffn_down`'s output at this layer to make the model produce
    /// the target token.
    ///
    /// Algorithm (first-order ROME, sidesteps the partial-forward path
    /// the iterative paper version requires):
    ///   1. Forward the subject prompt with full activation capture.
    ///   2. Compute logits + cross-entropy loss vs `target_token_id`
    ///      at the last position.
    ///   3. Backprop through output projection + final RMSNorm.
    ///   4. Walk backward through layers `n_layers−1 .. target_layer+1`
    ///      via existing `Forward::backward_step` with all LoRA slots
    ///      set to `None` and `backward_layer_floor = target_layer + 1`.
    ///   5. After step 4, the running `d_hidden` scratch buffer holds
    ///      `∂loss/∂hidden_input[target_layer + 1]`. By the residual
    ///      chain rule that equals `∂loss/∂ffn_out[target_layer]`
    ///      (plus `∂loss/∂attn_out[target_layer]` and
    ///      `∂loss/∂hidden_input[target_layer]`, but those don't
    ///      depend on what we're substituting).
    ///   6. Read back `d_hidden` as a `[d_model]` f32 vector.
    ///
    /// Caller composes v* = `ffn_out[target_layer, last_pos] − α · gradient`
    /// where `α` is the user-tuned step size. The resulting v* is then
    /// inserted into the rank-1 update `W' = W + (v* − W k*) k*ᵀ / s`.
    pub async fn compute_rome_gradient_native(
        &mut self,
        prompt_tokens: &[u32],
        target_layer: u32,
        target_token_id: u32,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            return Err(crate::error::RullamaError::Inference(
                "compute_rome_gradient_native: prompt_tokens must be non-empty".into(),
            ));
        }
        let n_layers = self.forward.cfg().n_layers;
        if target_layer >= n_layers {
            return Err(crate::error::RullamaError::Inference(format!(
                "compute_rome_gradient_native: target_layer {target_layer} out of range (have {n_layers})"
            )));
        }
        let ctx_arc = Arc::new(self.forward.ctx().clone());
        let cfg = self.forward.cfg().clone();
        let seq_len = prompt_tokens.len() as u32;
        let last_position = (prompt_tokens.len() - 1) as u32;

        // Forward with capture so backward has the activations it needs.
        let capture = crate::reference::rome::RomeCapture::new(&ctx_arc, &cfg, seq_len);
        let captures = capture.as_captures();
        self.forward.reset();
        for &tok in prompt_tokens {
            let _ = self
                .forward
                .step_capture(tok, &captures, None)
                .await?;
        }

        // Allocate backward scratch sized for this sequence length.
        let scratch = crate::reference::rome::RomeBackwardScratch::new(
            self.forward.ctx(),
            &cfg,
            seq_len,
        );
        let scratch_view = scratch.view();

        // All-None LoRA slots + grads — we don't have an adapter and
        // don't want to accumulate gradients into one.
        let empty_loras = crate::reference::rome::empty_lora_slots(n_layers);
        let empty_grads = crate::reference::rome::empty_lora_grads(n_layers);

        // Backward, stopping right above target_layer. The d_hidden
        // buffer in scratch_view will hold the residual-stream
        // gradient at hidden_input[target_layer+1] = (chain rule)
        // gradient w.r.t. ffn_out[target_layer].
        //
        // history_len = seq_len since we forwarded all tokens.
        // pos = last_position (the position we want the gradient at).
        // recompute_captures = false — captures from the forward
        //   above are still in the RomeCapture buffers.
        // backward_layer_floor = target_layer + 1 — stop the layer
        //   walk after processing layer target_layer + 1, leaving
        //   d_hidden as the gradient at that layer's input (=
        //   target_layer's output).
        let _loss = self
            .forward
            .backward_step_with_progress(
                target_token_id,
                &captures,
                &empty_loras,
                &empty_grads,
                &scratch_view,
                seq_len,
                last_position,
                false,
                None,
                target_layer + 1, // stop after layer target_layer+1
            )
            .await?;
        // Drop borrows before the readback's separate encoder
        // submission.
        drop(scratch_view);
        drop(captures);

        scratch.read_d_hidden(self.forward.ctx()).await
    }

    /// **ROME Phase 2.b — paper-faithful iterative v\* edit.**
    ///
    /// Mirrors kmeng01/rome's `compute_v.py` / `rome_main.py`: runs a
    /// 25-step Adam optimization over a residual-stream perturbation δ
    /// (shape `[d_model]`) injected at the SUBJECT-LAST token's
    /// position after `target_layer`. The objective per step is
    ///
    /// ```text
    ///   loss = -log P(target | prompt with hidden[L, subj_last] += δ)
    ///        + λ_wd · ‖δ‖² / ‖target_init‖²
    /// ```
    ///
    /// (KL preservation term is omitted in this iteration — Phase
    /// 2.b.3 adds it after this scaffolding lands.)
    ///
    /// After each Adam step, δ is L2-clamped to ‖δ‖ ≤ 4·‖target_init‖
    /// (paper's `clamp_norm_factor = 4`). At convergence,
    /// `v_star = target_init + δ_final` is the new MLP-output vector
    /// that, substituted at the subject-last position's `ffn_out[L]`,
    /// makes the model produce the target token.
    ///
    /// Returns the safetensors bytes for a rank-1 LoRA on
    /// `lora.blk.{L}.ffn_down.{A,B}` where
    ///   A = k\* (shape `[1, d_ffn]`)
    ///   B = δ_final / (k\*·k\*) (shape `[d_model, 1]`)
    ///
    /// This is the spherical-covariance formulation (`mom2_adjustment
    /// = false` per EasyEdit's Llama-3.2-3B config — the closest
    /// scale analog to Gemma 4 e2b).
    #[allow(clippy::too_many_arguments)]
    pub async fn rome_edit_iterative_native(
        &mut self,
        prompt_tokens: &[u32],
        subject_last_pos: u32,
        target_layer: u32,
        target_token_id: u32,
        hparams: RomeIterativeHparams,
    ) -> Result<Vec<u8>> {
        use crate::reference::forward_chained::RomeDeltaInjection;
        use crate::reference::rome::{
            RomeBackwardScratch, RomeCapture, RomeIterativeState, empty_lora_grads,
            empty_lora_slots,
        };

        if prompt_tokens.is_empty() {
            return Err(crate::error::RullamaError::Inference(
                "rome_edit_iterative_native: prompt_tokens must be non-empty".into(),
            ));
        }
        let n_layers = self.forward.cfg().n_layers;
        if target_layer >= n_layers {
            return Err(crate::error::RullamaError::Inference(format!(
                "rome_edit_iterative_native: target_layer {target_layer} out of range (have {n_layers})"
            )));
        }
        let seq_len = prompt_tokens.len() as u32;
        if subject_last_pos >= seq_len {
            return Err(crate::error::RullamaError::Inference(format!(
                "rome_edit_iterative_native: subject_last_pos {subject_last_pos} >= seq_len {seq_len}"
            )));
        }
        // Loss position: the LAST prompt token (where the model
        // predicts the next-token = target). For prompt
        // "What's the capital of France?", subject_last_pos = index
        // of "France" (where δ is injected), and loss_pos = index of
        // "?" (where target_token "Brie" should be predicted).
        let loss_pos = seq_len - 1;

        let ctx_arc = Arc::new(self.forward.ctx().clone());
        let cfg = self.forward.cfg().clone();
        let d_model = cfg.d_model;

        // Alloc-once GPU state. The capture buffers store activations
        // for backward; the scratch buffers store the backward state;
        // iter_state holds δ across iterations.
        let capture = RomeCapture::new(&ctx_arc, &cfg, seq_len);
        let scratch = RomeBackwardScratch::new(self.forward.ctx(), &cfg, seq_len);
        let iter_state = RomeIterativeState::new(&ctx_arc, d_model);
        let empty_loras = empty_lora_slots(n_layers);
        let empty_grads = empty_lora_grads(n_layers);

        // ---------- Step 0: clean forward to capture target_init and k* ----------
        {
            let captures = capture.as_captures();
            self.forward.reset();
            for &tok in prompt_tokens {
                let _ = self.forward.step_capture(tok, &captures, None).await?;
            }
        }
        let target_init = capture.read_ffn_out(target_layer, subject_last_pos).await?;
        let k_star = capture.read_ffn_act(target_layer, subject_last_pos).await?;
        let target_init_norm_sq: f32 = target_init.iter().map(|x| x * x).sum();
        let target_init_norm = target_init_norm_sq.sqrt();
        let k_norm_sq: f32 = k_star.iter().map(|x| x * x).sum();
        if k_norm_sq <= 1e-8 {
            return Err(crate::error::RullamaError::Inference(format!(
                "rome_edit_iterative_native: ||k*||² = {k_norm_sq:.3e} too small"
            )));
        }
        eprintln!(
            "[rome-iter] init: ||k*||²={:.3e} ||target_init||={:.3e} d_model={} d_ffn={}",
            k_norm_sq,
            target_init_norm,
            d_model,
            k_star.len()
        );

        // ---------- Adam state (CPU side; d_model=1536 is small) ----------
        let d = d_model as usize;
        let mut delta_cpu = vec![0.0f32; d];
        let mut m_cpu = vec![0.0f32; d];
        let mut v_cpu = vec![0.0f32; d];
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let max_norm = hparams.clamp_norm_factor * target_init_norm;

        let mut final_loss = f32::INFINITY;

        // ---------- Iterative loop ----------
        for it in 0..hparams.num_steps {
            // Edited forward: only the subject-last token uses the δ-injection path.
            {
                let captures = capture.as_captures();
                let rome_delta = RomeDeltaInjection {
                    delta_buf: &iter_state.delta,
                    target_layer,
                };
                self.forward.reset();
                for (i, &tok) in prompt_tokens.iter().enumerate() {
                    if i as u32 == subject_last_pos {
                        let _ = self
                            .forward
                            .step_capture_with_rome_delta(
                                tok,
                                &captures,
                                RomeDeltaInjection {
                                    delta_buf: rome_delta.delta_buf,
                                    target_layer,
                                },
                            )
                            .await?;
                    } else {
                        let _ = self.forward.step_capture(tok, &captures, None).await?;
                    }
                }
            }

            // Backward: gets ∂loss/∂hidden[target_layer+1] in scratch.d_hidden,
            // which by the residual chain rule equals ∂loss/∂δ (since δ feeds
            // straight into hidden at position subject_last_pos).
            //
            // Backward measures loss at loss_pos using target_token_id.
            let nll;
            {
                let scratch_view = scratch.view();
                nll = self
                    .forward
                    .backward_step_with_progress(
                        target_token_id,
                        &capture.as_captures(),
                        &empty_loras,
                        &empty_grads,
                        &scratch_view,
                        seq_len,
                        loss_pos,
                        false,
                        None,
                        target_layer + 1,
                    )
                    .await?;
            }
            let grad_at_layer_plus_1 = scratch.read_d_hidden(self.forward.ctx()).await?;
            // The gradient at hidden[L+1, loss_pos] is for the LOSS_POS row;
            // but δ is added at subject_last_pos. The backward path with
            // floor=L+1 reads from `d_hidden` which is the running grad on
            // the residual stream at the CURRENT decode position. Since
            // backward proceeds in reverse order of the FORWARD, and the
            // hidden buffer is overwritten per-position during forward,
            // `d_hidden` at the end of the layer walk holds the gradient
            // w.r.t. the residual stream at the SUBJECT_LAST position's
            // contribution at layer L+1 — which IS the gradient w.r.t. δ.
            // (See compute_rome_gradient_native; same structure.)
            let grad = grad_at_layer_plus_1;
            debug_assert_eq!(grad.len(), d);

            // Loss components
            let delta_norm_sq: f32 = delta_cpu.iter().map(|x| x * x).sum();
            let wd_loss = hparams.v_weight_decay * delta_norm_sq / target_init_norm_sq;
            let total_loss = nll + wd_loss;
            final_loss = total_loss;

            eprintln!(
                "[rome-iter {:>2}/{:>2}] nll={:.4e} wd={:.4e} loss={:.4e} ||δ||={:.3e}",
                it + 1,
                hparams.num_steps,
                nll,
                wd_loss,
                total_loss,
                delta_norm_sq.sqrt(),
            );

            // Early exit per the paper's `if loss < 5e-2: break`
            if total_loss < hparams.early_stop {
                eprintln!("[rome-iter] early stop: loss < {}", hparams.early_stop);
                break;
            }
            if it == hparams.num_steps - 1 {
                break; // skip the Adam update on the last iter
            }

            // CPU AdamW step on δ. Gradient also includes the weight-decay
            // derivative: ∂(λ‖δ‖²/‖t‖²) / ∂δ = 2λδ/‖t‖² ≈ wd · δ (folding 2 into wd).
            let t = (it + 1) as f32;
            let wd_grad_coef = hparams.v_weight_decay / target_init_norm_sq.max(1e-12);
            for i in 0..d {
                let g = grad[i] + wd_grad_coef * delta_cpu[i];
                m_cpu[i] = beta1 * m_cpu[i] + (1.0 - beta1) * g;
                v_cpu[i] = beta2 * v_cpu[i] + (1.0 - beta2) * g * g;
                let m_hat = m_cpu[i] / (1.0 - beta1.powf(t));
                let v_hat = v_cpu[i] / (1.0 - beta2.powf(t));
                delta_cpu[i] -= hparams.v_lr * m_hat / (v_hat.sqrt() + eps);
            }

            // Norm clamp: project δ to ‖δ‖ ≤ 4·‖target_init‖
            let dn_sq: f32 = delta_cpu.iter().map(|x| x * x).sum();
            let dn = dn_sq.sqrt();
            if dn > max_norm {
                let s = max_norm / dn;
                for x in delta_cpu.iter_mut() {
                    *x *= s;
                }
            }

            // Push updated δ to GPU for the next iteration's injection.
            iter_state.write_delta(&delta_cpu)?;
        }

        // ---------- Build the rank-1 adapter ----------
        // v_star = target_init + δ_final;  Δv = v_star − target_init = δ_final
        // (target_init is the unperturbed ffn_out[L, subj_last_pos] which IS W·k* in
        // the residual-stream formulation; so Δv = v_star − W·k* = δ.)
        //
        // Rank-1 form on ffn_down (weight [d_model × d_ffn]):
        //   A = k*                 shape [1, d_ffn]
        //   B = δ_final / (k*·k*)  shape [d_model, 1]
        //
        // At input = k*: LoRA contribution = (k*·k*/||k*||²) · δ = δ_final.
        let a_vals: Vec<f32> = k_star;
        let b_vals: Vec<f32> = delta_cpu.iter().map(|d| d / k_norm_sq).collect();

        eprintln!(
            "[rome-iter] final: loss={:.4e} ||δ||={:.3e} (capped at {:.3e})",
            final_loss,
            delta_cpu.iter().map(|x| x * x).sum::<f32>().sqrt(),
            max_norm
        );

        use safetensors::tensor::{Dtype, TensorView};
        let d_ffn = a_vals.len() as u32;
        let a_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec();
        let b_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec();
        let a_name = format!("lora.blk.{}.ffn_down.A", target_layer);
        let b_name = format!("lora.blk.{}.ffn_down.B", target_layer);
        let a_view = TensorView::new(Dtype::F32, vec![1usize, d_ffn as usize], &a_bytes)
            .map_err(|e| crate::error::RullamaError::Inference(format!("safetensors A: {e}")))?;
        let b_view = TensorView::new(Dtype::F32, vec![d_model as usize, 1usize], &b_bytes)
            .map_err(|e| crate::error::RullamaError::Inference(format!("safetensors B: {e}")))?;
        let mut views: std::collections::HashMap<&str, TensorView<'_>> =
            std::collections::HashMap::new();
        views.insert(a_name.as_str(), a_view);
        views.insert(b_name.as_str(), b_view);
        let metadata: std::collections::HashMap<String, String> = [
            ("format".to_string(), "rullama-lora-v0".to_string()),
            ("rank".to_string(), "1".to_string()),
            ("alpha".to_string(), "1.0".to_string()),
            ("target_modules".to_string(), "ffn_down".to_string()),
            ("dtype".to_string(), "f32".to_string()),
            ("rome".to_string(), "1".to_string()),
            ("rome_mode".to_string(), "iterative".to_string()),
            ("rome_layer".to_string(), target_layer.to_string()),
            ("rome_target_token".to_string(), target_token_id.to_string()),
            ("rome_subject_last_pos".to_string(), subject_last_pos.to_string()),
            ("rome_num_steps".to_string(), hparams.num_steps.to_string()),
            ("rome_v_lr".to_string(), hparams.v_lr.to_string()),
            ("rome_final_loss".to_string(), format!("{final_loss:.6e}")),
        ]
        .into_iter()
        .collect();
        safetensors::serialize(&views, &Some(metadata)).map_err(|e| {
            crate::error::RullamaError::Inference(format!("safetensors serialize: {e}"))
        })
    }

    /// **ROME Phase 1.3 — full edit pipeline.**
    ///
    /// Build a rank-1 LoRA adapter on `ffn_down` at `target_layer`
    /// that biases `ffn_down`'s output toward making the model
    /// produce `target_token_id` when asked the subject prompt.
    ///
    /// Returns the safetensors bytes (compatible with the existing
    /// `load_adapter_native` path). Caller writes to OPFS / local fs
    /// however they want.
    ///
    /// Math:
    ///   k* = `extract_mlp_input_native(prompt, target_layer)`  — `[d_ffn]`
    ///   g  = `compute_rome_gradient_native(prompt, target_layer, target)` — `[d_model]`
    ///   v_star_delta = -alpha * g                              — `[d_model]`
    ///
    /// Rank-1 LoRA on ffn_down (weight shape `[d_model × d_ffn]`):
    ///   A = k*   shape `[1, d_ffn]`
    ///   B = v_star_delta   shape `[d_model, 1]`
    ///   metadata alpha = 1 (so LoRA's `scale = 1/rank = 1`)
    ///
    /// LoRA forward correction is `scale · B · (A · input)`. When
    /// `input = k*`, `A · input = ||k*||²`, so the correction is
    /// `||k*||² · v_star_delta`. The effective edit magnitude
    /// therefore scales with `||k*||²` × `alpha` — caller tunes
    /// `alpha` to control edit strength. Different layers have
    /// different typical `||k*||²` so per-layer tuning is expected
    /// (see Phase 1.5 sweep).
    pub async fn rome_edit_native(
        &mut self,
        prompt_tokens: &[u32],
        target_layer: u32,
        target_token_id: u32,
        alpha: f32,
    ) -> Result<Vec<u8>> {
        self.rome_edit_native_inner(prompt_tokens, target_layer, target_token_id, alpha, None)
            .await
    }

    /// **ROME Phase 2.3 — covariance-corrected edit.**
    ///
    /// Same shape as [`Model::rome_edit_native`] but replaces the
    /// spherical denominator `s = ||k*||²` with the covariance-weighted
    /// `s = k*ᵀ C⁻¹ k*` where `C = E[k kᵀ]` is the typical-key
    /// covariance precomputed by `examples/compute_rome_covariance.rs`
    /// and loaded via [`reference::rome::RomeCovariance`].
    ///
    /// The LoRA A factor becomes `(C⁻¹ k*) / s` (rather than just `k*`)
    /// so that at typical inputs `x ~ C` the contribution
    /// `((C⁻¹ k*)ᵀ x / s) · Δv` averages to zero, preserving unrelated
    /// facts. At `x = k*` exactly the contribution is `Δv` (because
    /// `(C⁻¹ k*)ᵀ k* / s = 1`), so the targeted edit fires unchanged.
    ///
    /// `alpha` retains the "step size in gradient-direction space"
    /// interpretation from the spherical version — `Δv = -α · g`.
    pub async fn rome_edit_native_with_covariance(
        &mut self,
        prompt_tokens: &[u32],
        target_layer: u32,
        target_token_id: u32,
        alpha: f32,
        covariance: &crate::reference::rome::RomeCovariance,
    ) -> Result<Vec<u8>> {
        if !covariance.has_layer(target_layer) {
            return Err(crate::error::RullamaError::Inference(format!(
                "rome_edit_native_with_covariance: sidecar has no factor for layer {target_layer} \
                 (have layers {:?})",
                covariance.layers()
            )));
        }
        self.rome_edit_native_inner(
            prompt_tokens,
            target_layer,
            target_token_id,
            alpha,
            Some(covariance),
        )
        .await
    }

    /// Shared implementation. `covariance = None` → spherical
    /// (ROME-lite); `Some` → full ROME with `C⁻¹` scaling.
    async fn rome_edit_native_inner(
        &mut self,
        prompt_tokens: &[u32],
        target_layer: u32,
        target_token_id: u32,
        alpha: f32,
        covariance: Option<&crate::reference::rome::RomeCovariance>,
    ) -> Result<Vec<u8>> {
        // Step A: k* extraction. After this, KV is populated with
        // the subject prompt; `extract_mlp_input_native` runs a
        // forward with capture and reads back ffn_act at the last
        // position.
        let k_star = self
            .extract_mlp_input_native(prompt_tokens, target_layer)
            .await?;

        // Step B: v* gradient. This re-runs the forward (resetting
        // KV) so the captures match what backward expects. The
        // running d_hidden after backward stops at target_layer+1
        // gives ∂loss/∂ffn_out[target_layer].
        let grad = self
            .compute_rome_gradient_native(prompt_tokens, target_layer, target_token_id)
            .await?;

        // Step C: form rank-1 LoRA.
        //
        // Spherical (ROME-lite, C ≈ I):
        //   A = k*
        //   B = -α · g / ||k*||²
        //   At x = k*: contribution = (k*·k*/||k*||²) · (-α g) = -α g
        //
        // Covariance-corrected (full ROME):
        //   u = C⁻¹ k*
        //   s = k*·u   (= k*ᵀ C⁻¹ k*)
        //   A = u / s
        //   B = -α · g
        //   At x = k*: contribution = ((u/s)·k*) · (-α g) = (s/s)·(-α g) = -α g
        //   At x ~ C: E[(u/s)·x] ≈ 0  (the C-orthogonal property)
        //
        // The full-ROME form is the one that preserves unrelated facts
        // — the spherical form leaks because typical k vectors aren't
        // orthogonal to k* in the L2 metric, but they ARE on average
        // C-orthogonal under the proper inner product.
        let d_ffn = k_star.len() as u32;
        let d_model = grad.len() as u32;

        let (a_vals, b_vals, scale_hint) = if let Some(cov) = covariance {
            let u: Vec<f32> = cov.cov_inv_k(target_layer, &k_star)?;
            let s: f32 = k_star.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            if !s.is_finite() || s.abs() < 1e-8 {
                return Err(crate::error::RullamaError::Inference(format!(
                    "rome_edit_native_with_covariance: k*ᵀ C⁻¹ k* = {s:.3e} \
                     too small to invert (raise ridge during calibration?)"
                )));
            }
            eprintln!(
                "[rome] mode=covariance layer={target_layer} \
                 ||k*||²={:.3e} s=k*ᵀC⁻¹k*={:.3e} alpha={}",
                k_star.iter().map(|&x| x * x).sum::<f32>(),
                s,
                alpha
            );
            let inv_s = 1.0_f32 / s;
            let a: Vec<f32> = u.iter().map(|&x| x * inv_s).collect();
            let b: Vec<f32> = grad.iter().map(|&g| -alpha * g).collect();
            (a, b, s)
        } else {
            let k_norm_sq: f32 = k_star.iter().map(|&x| x * x).sum();
            if k_norm_sq <= 1e-8 {
                return Err(crate::error::RullamaError::Inference(format!(
                    "rome_edit_native: ||k*||² = {k_norm_sq:.3e} too small to invert"
                )));
            }
            eprintln!(
                "[rome] mode=spherical layer={target_layer} \
                 ||k*||²={:.3e} alpha={}",
                k_norm_sq, alpha
            );
            let scale = -alpha / k_norm_sq;
            let b: Vec<f32> = grad.iter().map(|&g| scale * g).collect();
            (k_star.clone(), b, k_norm_sq)
        };

        // Step D: serialize as safetensors with the same tensor-name
        // + metadata conventions `rullama_finetune::TrainingSession::
        // save_adapter_to_bytes` uses. The existing
        // `lora::InferenceAdapter::from_safetensors_bytes` parses it
        // identically.
        use safetensors::tensor::{Dtype, TensorView};
        let a_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&a_vals).to_vec();
        let b_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&b_vals).to_vec();
        let a_name = format!("lora.blk.{}.ffn_down.A", target_layer);
        let b_name = format!("lora.blk.{}.ffn_down.B", target_layer);
        let a_view = TensorView::new(Dtype::F32, vec![1usize, d_ffn as usize], &a_bytes)
            .map_err(|e| crate::error::RullamaError::Inference(format!("safetensors A: {e}")))?;
        let b_view = TensorView::new(Dtype::F32, vec![d_model as usize, 1usize], &b_bytes)
            .map_err(|e| crate::error::RullamaError::Inference(format!("safetensors B: {e}")))?;
        let mut views: std::collections::HashMap<&str, TensorView<'_>> =
            std::collections::HashMap::new();
        views.insert(a_name.as_str(), a_view);
        views.insert(b_name.as_str(), b_view);
        let mode = if covariance.is_some() {
            "covariance"
        } else {
            "spherical"
        };
        let metadata: std::collections::HashMap<String, String> = [
            ("format".to_string(), "rullama-lora-v0".to_string()),
            ("rank".to_string(), "1".to_string()),
            ("alpha".to_string(), "1.0".to_string()),
            ("target_modules".to_string(), "ffn_down".to_string()),
            ("dtype".to_string(), "f32".to_string()),
            ("rome".to_string(), "1".to_string()),
            ("rome_layer".to_string(), target_layer.to_string()),
            ("rome_alpha".to_string(), alpha.to_string()),
            ("rome_target_token".to_string(), target_token_id.to_string()),
            ("rome_mode".to_string(), mode.to_string()),
            ("rome_scale".to_string(), format!("{:.6e}", scale_hint)),
        ]
        .into_iter()
        .collect();
        safetensors::serialize(&views, &Some(metadata)).map_err(|e| {
            crate::error::RullamaError::Inference(format!("safetensors serialize: {e}"))
        })
    }

    /// Feed one position with a pre-computed `[d_model]` embedding instead of a
    /// token id — the path multimodal soft tokens take (each row of the
    /// `encode_image` / `encode_audio` output is one such embedding). Returns the
    /// sampled next token id, just like `step_native`. The sampler is *not* given
    /// an "observed token" — soft tokens have no id to penalise.
    pub async fn step_with_embedding_native(&mut self, embedding: &[f32]) -> Result<u32> {
        // Mirror the adapter routing in `step_native` so multimodal
        // soft-token steps respect a loaded LoRA adapter. Without
        // this, image/audio prefill silently bypasses the adapter
        // even though the matching text steps honour it.
        let logits = match &self.adapter {
            Some(adapter) => {
                let slots = adapter.layer_slots(self.forward.cfg().n_layers);
                self.forward
                    .step_with_embedding_with_lora(embedding, &slots)
                    .await?
            }
            None => self.forward.step_with_embedding(embedding).await?,
        };
        let next = self.sampler.sample(&logits);
        Ok(next)
    }

    /// Render a list of chat messages into the Gemma 4 prompt format, ready to feed
    /// to `encode_tokens` + `step`. Includes the trailing `<|turn>model\n` so the
    /// next sampled token starts the assistant reply.
    pub fn render_chat_native(&self, messages: &[ChatMessage], with_bos: bool) -> String {
        gemma4_small::render_for_completion(messages, with_bos)
    }

    /// Like [`render_chat_native`] but leaves a trailing assistant turn
    /// open if the last message has `role: Model`. Used by suspend/resume
    /// when rebuilding KV from a conversation that already contains a
    /// partial assistant response — the model continues *that* response
    /// rather than starting a new one.
    pub fn render_chat_for_continuation_native(
        &self,
        messages: &[ChatMessage],
        with_bos: bool,
    ) -> String {
        gemma4_small::render_for_continuation(messages, with_bos)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Model {
    /// JS entry point: build a Model from raw GGUF bytes (e.g. a `Uint8Array` from
    /// `fetch().then(r => r.arrayBuffer())`). Holds the entire GGUF in wasm linear
    /// memory; only suitable for files that fit under the 4 GB wasm32 cap.
    #[wasm_bindgen(js_name = load)]
    pub async fn load_js(bytes: Vec<u8>) -> std::result::Result<Model, JsError> {
        Self::load_native(bytes)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// JS entry point: stream the GGUF over HTTP via byte-range requests. The full
    /// file never lands in wasm memory — tensors are fetched on demand and dropped
    /// after each GPU upload. This is the path that lets `gemma4:e2b` (~7 GB) load
    /// in the browser despite wasm32's 4 GB linear-memory cap.
    ///
    /// Requires the server to support `Range: bytes=N-M` and to expose either
    /// `Content-Range` or `X-Total-Size` so the client can discover the file length.
    #[wasm_bindgen(js_name = loadFromUrl)]
    pub async fn load_from_url_js(url: String) -> std::result::Result<Model, JsError> {
        let fetcher = crate::gguf::HttpRangeFetcher::new(url)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))?;
        let arc: std::sync::Arc<dyn crate::gguf::TensorFetcher> = std::sync::Arc::new(fetcher);
        Self::load_streaming(arc)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// JS entry point: stream the GGUF from a file the host has already saved to
    /// OPFS (Origin Private File System). `read_fn` is a JS callback with signature
    /// `(offset_f64, len_f64) -> Promise<Uint8Array> | Uint8Array`. `total_bytes`
    /// is the file's full size (caller knows this from the OPFS file handle).
    ///
    /// This is the path that bypasses iOS Safari's ~5.6 GiB single-Blob cap and
    /// ~2 GiB live-JS-heap cap — bytes are read directly from the disk-backed
    /// OPFS file in slices and never aggregate in JS memory.
    /// JS entry point: stream the GGUF from an OPFS-resident file with
    /// vision + audio towers built. Optional `max_context` caps the KV
    /// pre-allocation; pass 0 to use the compile-time `MAX_CONTEXT`
    /// (4096). On iPhone, supplying 2048 saves ~600 MB of KV buffer
    /// against the multimodal weight budget.
    #[wasm_bindgen(js_name = loadFromOpfs)]
    pub async fn load_from_opfs_js(
        read_fn: js_sys::Function,
        total_bytes: f64,
        max_context: u32,
    ) -> std::result::Result<Model, JsError> {
        if !total_bytes.is_finite() || total_bytes < 0.0 {
            return Err(JsError::new(
                "loadFromOpfs: total_bytes must be a non-negative finite number",
            ));
        }
        let total = total_bytes as u64;
        let fetcher = crate::gguf::OpfsFetcher::new(read_fn, total);
        let arc: std::sync::Arc<dyn crate::gguf::TensorFetcher> = std::sync::Arc::new(fetcher);
        Self::load_streaming_with_max_context(arc, max_context)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// JS entry point: text-only variant of [`loadFromOpfs`]. Skips vision and
    /// audio tower construction AND caps the KV cache at `max_context` tokens
    /// (default 512 if `max_context` is 0 or absent) so the wasm-load
    /// footprint stays small enough to fit a Q4_K_M `gemma4:e2b` in
    /// iPhone-class shared RAM (8 GB). `encode_image` / `encode_audio` will
    /// fail with "this checkpoint has no vision/audio tower" — text
    /// inference and chat work as normal.
    #[wasm_bindgen(js_name = loadFromOpfsTextOnly)]
    pub async fn load_from_opfs_text_only_js(
        read_fn: js_sys::Function,
        total_bytes: f64,
        max_context: u32,
    ) -> std::result::Result<Model, JsError> {
        if !total_bytes.is_finite() || total_bytes < 0.0 {
            return Err(JsError::new(
                "loadFromOpfsTextOnly: total_bytes must be a non-negative finite number",
            ));
        }
        let total = total_bytes as u64;
        let max_ctx = if max_context == 0 { 512 } else { max_context };
        let fetcher = crate::gguf::OpfsFetcher::new(read_fn, total);
        let arc: std::sync::Arc<dyn crate::gguf::TensorFetcher> = std::sync::Arc::new(fetcher);
        Self::load_streaming_text_only(arc, max_ctx)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    #[wasm_bindgen(js_name = encode)]
    pub fn encode_js(&self, text: &str) -> Vec<u32> {
        self.encode_tokens(text)
    }

    #[wasm_bindgen(js_name = tokenStr)]
    pub fn token_str_js(&self, id: u32) -> Option<String> {
        self.token_str_native(id)
    }

    #[wasm_bindgen(js_name = vocabSize, getter)]
    pub fn vocab_size_js(&self) -> u32 {
        self.vocab_size_native()
    }

    #[wasm_bindgen(js_name = position, getter)]
    pub fn position_js(&self) -> u32 {
        self.position_native()
    }

    #[wasm_bindgen(js_name = isEos)]
    pub fn is_eos_js(&self, id: u32) -> bool {
        self.is_eos_native(id)
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&mut self) {
        self.reset_native()
    }

    /// Snapshot KV cache + sampler state into a single Uint8Array. Caller
    /// writes the result to OPFS / IndexedDB for suspend/resume.
    #[wasm_bindgen(js_name = saveKvState)]
    pub async fn save_kv_state_js(&self) -> std::result::Result<Vec<u8>, JsError> {
        self.save_kv_state_native()
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Inverse of [`saveKvState`]. Validates the snapshot against the
    /// currently-loaded model (layout hash) and refuses to apply if it's
    /// from a different model architecture — caller should fall back to
    /// token-replay rebuild in that case.
    #[wasm_bindgen(js_name = restoreKvState)]
    pub fn restore_kv_state_js(&mut self, bytes: Vec<u8>) -> std::result::Result<(), JsError> {
        self.restore_kv_state_native(&bytes)
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Feed one token, advance pos, return sampled next token id.
    #[wasm_bindgen(js_name = step)]
    pub async fn step_js(&mut self, token_id: u32) -> std::result::Result<u32, JsError> {
        self.step_native(token_id)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Feed one pre-computed embedding (e.g. one soft-token row from
    /// `encodeImage`), advance pos, return sampled next token id. JS pass-in is a
    /// `Float32Array` of length `d_model` (1536 for gemma4:e2b).
    #[wasm_bindgen(js_name = stepWithEmbedding)]
    pub async fn step_with_embedding_js(
        &mut self,
        embedding: Vec<f32>,
    ) -> std::result::Result<u32, JsError> {
        self.step_with_embedding_native(&embedding)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Configure sampling from a JSON-shape `{temperature, top_k, top_p, repetition_penalty, seed}`.
    /// JS callers pass an object; serde decodes it.
    #[wasm_bindgen(js_name = setSampling)]
    pub fn set_sampling_js(&mut self, opts_json: JsValue) -> std::result::Result<(), JsError> {
        let opts: SamplingOptions = serde_wasm_bindgen::from_value(opts_json)
            .map_err(|e| JsError::new(&format!("invalid sampling options: {e}")))?;
        self.sampler.set_options(opts);
        Ok(())
    }

    /// True iff a LoRA adapter is currently active.
    #[wasm_bindgen(js_name = hasAdapter, getter)]
    pub fn has_adapter_js(&self) -> bool {
        self.has_adapter_native()
    }

    /// Load a safetensors LoRA adapter from raw bytes (e.g. the
    /// `Uint8Array` returned by `TrainingSession.saveAdapter`).
    /// Returns the number of LoRA slots loaded.
    #[wasm_bindgen(js_name = loadAdapter)]
    pub fn load_adapter_js(&mut self, bytes: Vec<u8>) -> std::result::Result<usize, JsError> {
        self.load_adapter_native(&bytes)
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Drop the active adapter.
    #[wasm_bindgen(js_name = clearAdapter)]
    pub fn clear_adapter_js(&mut self) {
        self.clear_adapter_native()
    }

    /// True iff this checkpoint carries a vision tower (gemma4:e2b/e4b).
    #[wasm_bindgen(js_name = hasVision, getter)]
    pub fn has_vision_js(&self) -> bool {
        self.has_vision_native()
    }

    /// Encode an RGB image into a `Float32Array` of soft-token embeddings, flat
    /// `[n_pooled_patches × d_text]`. JS pass-in: `pixels` is the image in
    /// channel-first `[R..., G..., B...]` order normalised to `[-1, 1]`; `h`,
    /// `w` are integer pixel dims aligned to `patch_size * n_merge` (= 48).
    #[wasm_bindgen(js_name = encodeImage)]
    pub async fn encode_image_js(
        &mut self,
        pixels: Vec<f32>,
        h: u32,
        w: u32,
        progress_cb: Option<js_sys::Function>,
    ) -> std::result::Result<Vec<f32>, JsError> {
        // Wrap the optional JS callback as a Rust closure that gets
        // called after each transformer layer; lets the UI show
        // "Analyzing image (N/M)…" instead of a frozen spinner.
        let cb: Option<Box<dyn Fn(u32, u32)>> = progress_cb.map(|f| {
            Box::new(move |layer: u32, total: u32| {
                let _ = f.call2(&JsValue::NULL, &JsValue::from(layer), &JsValue::from(total));
            }) as Box<dyn Fn(u32, u32)>
        });
        self.encode_image_native(&pixels, h as usize, w as usize, cb.as_deref())
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Number of soft tokens an `h × w` image will produce, or `null` if either
    /// dimension is misaligned.
    #[wasm_bindgen(js_name = imageSoftTokenCount)]
    pub fn image_soft_token_count_js(&self, h: u32, w: u32) -> Option<u32> {
        self.image_soft_token_count_native(h as usize, w as usize)
            .map(|n| n as u32)
    }

    /// `[<|image> token id, <image|> token id]` if both sentinels exist in the
    /// vocab, else `null`. Used by the JS chat handler to splice soft-token
    /// embeddings between the markers in the encoded prompt.
    #[wasm_bindgen(js_name = imageSentinelIds)]
    pub fn image_sentinel_ids_js(&self) -> Option<Vec<u32>> {
        let begin = self.tokenizer.str_to_id("<|image>")?;
        let end = self.tokenizer.str_to_id("<image|>")?;
        Some(vec![begin, end])
    }

    /// True iff this checkpoint carries an audio tower.
    #[wasm_bindgen(js_name = hasAudio, getter)]
    pub fn has_audio_js(&self) -> bool {
        self.has_audio_native()
    }

    /// Encode raw 16 kHz mono PCM (Float32Array in `[-1, 1]`) into a
    /// Float32Array of soft-token embeddings. Caller is responsible for
    /// resampling to 16 kHz if the source is at a different rate.
    #[wasm_bindgen(js_name = encodeAudio)]
    pub async fn encode_audio_js(
        &mut self,
        pcm: Vec<f32>,
    ) -> std::result::Result<Vec<f32>, JsError> {
        self.encode_audio_native(&pcm)
            .await
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Decode WAV file bytes into 16 kHz mono Float32Array. Convenience for JS
    /// callers that have a WAV file but don't want to plumb Web Audio.
    #[wasm_bindgen(js_name = decodeWav)]
    pub fn decode_wav_js(bytes: Vec<u8>) -> std::result::Result<Vec<f32>, JsError> {
        Self::decode_wav_native(&bytes).map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Cooperatively cancel an in-flight `encodeImage` / `encodeAudio`. The
    /// in-flight `Promise` rejects with a "cancelled" error on the next
    /// transformer-layer boundary (≤500 ms in practice). Safe to call when
    /// no encode is running — the flag is cleared at the start of the
    /// next encode regardless.
    #[wasm_bindgen(js_name = cancelMultimodalEncode)]
    pub fn cancel_multimodal_encode_js(&self) {
        self.cancel_multimodal_encode_native();
    }

    /// `[<|audio> token id, <audio|> token id]` if both sentinels exist; else `null`.
    #[wasm_bindgen(js_name = audioSentinelIds)]
    pub fn audio_sentinel_ids_js(&self) -> Option<Vec<u32>> {
        let begin = self.tokenizer.str_to_id("<|audio>")?;
        let end = self.tokenizer.str_to_id("<audio|>")?;
        Some(vec![begin, end])
    }

    /// Evict cached vision-tower weights from GPU memory. Returns the number
    /// of cache entries freed. Call between turns on iPhone when the next
    /// message won't include an image to free ~3 GB.
    #[wasm_bindgen(js_name = releaseVisionWeights)]
    pub fn release_vision_weights_js(&mut self) -> usize {
        self.release_vision_weights_native()
    }

    /// Evict cached audio-tower weights from GPU memory.
    #[wasm_bindgen(js_name = releaseAudioWeights)]
    pub fn release_audio_weights_js(&mut self) -> usize {
        self.release_audio_weights_native()
    }

    /// Re-allocate the per-layer KV cache at a new capacity (tokens).
    /// Returns the previous max_context so JS can restore later. See
    /// `shrink_kv_native` for the full rationale.
    #[wasm_bindgen(js_name = shrinkKv)]
    pub fn shrink_kv_js(&mut self, new_max_context: u32) -> std::result::Result<u32, JsValue> {
        self.shrink_kv_native(new_max_context)
            .map_err(|e| JsValue::from_str(&format!("{e}")))
    }

    /// Current KV cache capacity (tokens). Snapshot this before
    /// `shrinkKv()` and pass it back on `trainingFinish` to restore.
    #[wasm_bindgen(js_name = maxContext, getter)]
    pub fn max_context_js(&self) -> u32 {
        self.max_context_native()
    }

    /// Total bytes currently held in the shared GPU weight cache.
    #[wasm_bindgen(js_name = cachedWeightBytes, getter)]
    pub fn cached_weight_bytes_js(&self) -> u64 {
        self.cached_weight_bytes_native()
    }

    /// Render a single user message (and optional system message) into the Gemma 4
    /// chat-template prompt. JS callers pass `[{role, content}, ...]` as JSON.
    #[wasm_bindgen(js_name = renderChat)]
    pub fn render_chat_js(
        &self,
        messages_json: JsValue,
        with_bos: bool,
    ) -> std::result::Result<String, JsError> {
        let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages_json)
            .map_err(|e| JsError::new(&format!("invalid messages: {e}")))?;
        Ok(self.render_chat_native(&msgs, with_bos))
    }

    /// Like [`renderChat`] but leaves a trailing assistant turn OPEN if
    /// the last message has `role: "model"`. Used by suspend/resume to
    /// rebuild KV cache from a conversation that includes a partial
    /// assistant response.
    #[wasm_bindgen(js_name = renderChatForContinuation)]
    pub fn render_chat_for_continuation_js(
        &self,
        messages_json: JsValue,
        with_bos: bool,
    ) -> std::result::Result<String, JsError> {
        let msgs: Vec<ChatMessage> = serde_wasm_bindgen::from_value(messages_json)
            .map_err(|e| JsError::new(&format!("invalid messages: {e}")))?;
        Ok(self.render_chat_for_continuation_native(&msgs, with_bos))
    }
}

// ---------- (legacy) options shapes — retained from M0 stub for future use ----------

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

fn default_max_tokens() -> u32 {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> u32 {
    40
}
fn default_repetition_penalty() -> f32 {
    1.0
}
