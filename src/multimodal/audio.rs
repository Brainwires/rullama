//! Gemma 4 audio tower (CPU oracle): waveform → soft tokens.
//!
//! Mirrors Ollama's `model/models/gemma4/model_audio.go::AudioModel.ForwardAudio`
//! op-for-op. Pure Rust f32 — audio encoding is one-shot per chat turn (~tens
//! to a few hundred frames; ~12 Conformer blocks), so the few seconds of CPU
//! work per turn is acceptable for a v1 demo. Once parity vs Ollama is locked
//! we can port piece by piece to GPU; the CPU oracle stays as the regression
//! anchor (same role `forward.rs` plays for the text path).
//!
//! Pipeline:
//!   raw PCM 16 kHz mono
//!     → log-mel spectrogram [n_frames, 128]
//!     → SSCP: 2× Conv2D 3×3 stride 2 + LayerNorm + ReLU
//!     → linear projection to hidden_size (1024 for gemma4:e2b)
//!     → 12× Conformer block:
//!         · FFW start (half-residual): clamp → norm → up → SiLU → down → clamp → post_norm → x = res + 0.5*y
//!         · self-attention: chunked block-local with relative-position bias,
//!           per-dim Q scale, logit softcap=50
//!         · LightConv: norm → conv_pw1 → GLU → depthwise conv1d (kernel=5)
//!           → norm → SiLU → conv_pw2 → residual
//!         · FFW end (half-residual)
//!         · clamp(±1e10) → final RMSNorm
//!     → optional output_proj (linear with bias; absent in gemma4:e2b — skipped)
//!     → AudioMultimodalProjector: FC + bias → unweighted RMSNorm →
//!       ClippableLinear input_projection
//!   = soft-token embeddings [n_audio_tokens, d_text=1536]
//!
//! Constants pinned to Ollama's `newAudioModelOptions` — most aren't in the
//! GGUF, they're hard-coded in Ollama (chunk/past/future, softcap, etc.).

use std::sync::Arc;

use crate::backend::WeightCache;
use crate::error::{Result, RullamaError};
use crate::gguf::{GgufReader, dequant_tensor_to_f32_async};

use super::audio_features::{MelEngine, MEL_BINS};

#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub n_layers:      u32,    // gemma4.audio.block_count                (12)
    pub hidden:        u32,    // gemma4.audio.embedding_length           (1024)
    pub ffn_inter:     u32,    // gemma4.audio.feed_forward_length        (4096)
    pub n_heads:       u32,    // gemma4.audio.attention.head_count       (8)
    pub conv_kernel:   u32,    // gemma4.audio.conv_kernel_size           (5)
    pub mel_bins:      u32,    // 128 (also our MEL_BINS)
    pub eps:           f32,    // 1e-6 default
    pub chunk_size:    u32,    // 12  (Ollama-hardcoded)
    pub max_past:      u32,    // 12
    pub max_future:    u32,    // 0
    pub context_size:  u32,    // chunk_size + max_past + max_future = 24
    pub logit_cap:     f32,    // 50.0
    pub residual_w:    f32,    // 0.5
    pub grad_clip:     f32,    // 1e10
    pub d_text:        u32,    // text d_model — projector output
}

impl AudioConfig {
    pub fn from_gguf(r: &GgufReader, d_text: u32) -> Result<Self> {
        let n_layers = r.get_opt("gemma4.audio.block_count")
            .and_then(|v| v.as_u32().ok())
            .ok_or_else(|| RullamaError::Inference(
                "gemma4.audio.block_count missing — not a multimodal-audio GGUF?".into()
            ))?;
        let hidden    = r.get("gemma4.audio.embedding_length")?.as_u32()?;
        let ffn_inter = r.get("gemma4.audio.feed_forward_length")?.as_u32()?;
        let n_heads   = r.get("gemma4.audio.attention.head_count")?.as_u32()?;
        let conv_kernel = r.get_opt("gemma4.audio.conv_kernel_size")
            .and_then(|v| v.as_u32().ok()).unwrap_or(5);
        let mel_bins  = r.get_opt("gemma4.audio.num_mel_bins")
            .and_then(|v| v.as_u32().ok()).unwrap_or(MEL_BINS as u32);
        let eps = r.get_opt("gemma4.audio.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32().ok()).unwrap_or(1e-6);
        // The chunk/past/future/cap/residual constants live in Ollama's Go (not the GGUF).
        let chunk_size = 12;
        let max_past = 12;
        let max_future = 0;
        Ok(Self {
            n_layers, hidden, ffn_inter, n_heads, conv_kernel, mel_bins, eps,
            chunk_size, max_past, max_future,
            context_size: chunk_size + max_past + max_future,
            logit_cap: 50.0,
            residual_w: 0.5,
            grad_clip: 1e10,
            d_text,
        })
    }

    pub fn head_dim(&self) -> u32 { self.hidden / self.n_heads }
}

/// One Conformer block's resident weights, fully dequantised once at construction.
struct AudioBlock {
    pre_norm: Vec<f32>,
    // FFW start
    ffw_norm: Vec<f32>,
    ffw_up_w: Vec<f32>,    // [hidden, ffn_inter]
    ffw_up_b: Option<Vec<f32>>,
    ffw_down_w: Vec<f32>,  // [ffn_inter, hidden]
    ffw_down_b: Option<Vec<f32>>,
    ffw_post_norm: Vec<f32>,
    // FFW end
    ffw_norm_1: Vec<f32>,
    ffw_up_1_w: Vec<f32>,
    ffw_up_1_b: Option<Vec<f32>>,
    ffw_down_1_w: Vec<f32>,
    ffw_down_1_b: Option<Vec<f32>>,
    ffw_post_norm_1: Vec<f32>,
    // Attention
    attn_pre_norm: Vec<f32>,
    attn_post_norm: Vec<f32>,
    attn_q_w: Vec<f32>,
    attn_q_b: Option<Vec<f32>>,
    attn_k_w: Vec<f32>,
    attn_k_b: Option<Vec<f32>>,
    attn_v_w: Vec<f32>,
    attn_v_b: Option<Vec<f32>>,
    attn_o_w: Vec<f32>,
    attn_o_b: Option<Vec<f32>>,
    linear_pos: Vec<f32>,         // [hidden, n_heads * head_dim] — projects sinusoidal pos embeddings
    per_dim_scale: Vec<f32>,      // [head_dim] — already softplus'd by Ollama's converter
    // LightConv
    conv_norm: Vec<f32>,
    norm_conv: Vec<f32>,
    conv_pw1_w: Vec<f32>,    // hidden -> 2*hidden (GLU pre-split)
    conv_pw1_b: Option<Vec<f32>>,
    conv_pw2_w: Vec<f32>,    // hidden -> hidden
    conv_pw2_b: Option<Vec<f32>>,
    conv_dw: Vec<f32>,       // [hidden, kernel] depthwise conv weights

    // ClippableLinear clamps. Each `Clamp` is `{in_min, in_max, out_min, out_max}`
    // — defaults to all-zeros when the GGUF doesn't ship the scalar. Ollama's
    // `AudioClippableLinear.Forward` applies the input clamp iff `inMax != 0`
    // and the output clamp iff `outMax != 0` (default zero behaves as "skip").
    cl_attn_q: Clamp,
    cl_attn_k: Clamp,
    cl_attn_v: Clamp,
    cl_attn_o: Clamp,
    cl_ffw_up: Clamp,
    cl_ffw_down: Clamp,
    cl_ffw_up_1: Clamp,
    cl_ffw_down_1: Clamp,
    cl_conv_pw1: Clamp,
    cl_conv_pw2: Clamp,
}

#[derive(Clone, Copy, Default)]
struct Clamp { in_min: f32, in_max: f32, out_min: f32, out_max: f32 }

impl Clamp {
    fn apply_in(&self, x: &mut [f32]) {
        if self.in_max != 0.0 {
            for v in x { *v = v.clamp(self.in_min, self.in_max); }
        }
    }
    fn apply_out(&self, y: &mut [f32]) {
        if self.out_max != 0.0 {
            for v in y { *v = v.clamp(self.out_min, self.out_max); }
        }
    }
}

async fn load_clamp(r: &GgufReader, prefix: &str) -> Result<Clamp> {
    async fn one(r: &GgufReader, name: &str) -> Result<f32> {
        Ok(load_opt_f32(r, name).await?
            .and_then(|x| x.first().copied())
            .unwrap_or(0.0))
    }
    Ok(Clamp {
        in_min:  one(r, &format!("{prefix}.input_min")).await?,
        in_max:  one(r, &format!("{prefix}.input_max")).await?,
        out_min: one(r, &format!("{prefix}.output_min")).await?,
        out_max: one(r, &format!("{prefix}.output_max")).await?,
    })
}

/// CPU-resident audio tower. Construct once, reuse for every `encode` call.
pub struct AudioForward {
    cfg: AudioConfig,
    mel: MelEngine,

    // SSCP weights (2 × Conv2D + LayerNorm + linear projection).
    sscp0_w: Vec<f32>,        // [out_C0, in_C=1, kH=3, kW=3]
    sscp0_norm_w: Vec<f32>,
    sscp0_norm_b: Option<Vec<f32>>,
    sscp1_w: Vec<f32>,        // [out_C1, in_C=out_C0, kH=3, kW=3]
    sscp1_norm_w: Vec<f32>,
    sscp1_norm_b: Option<Vec<f32>>,
    pre_encode_out_w: Vec<f32>,  // linear: out_C1 * F'' → hidden
    pre_encode_out_b: Option<Vec<f32>>,
    sscp0_out_c: usize,
    sscp1_out_c: usize,
    sscp_proj_in: usize,

    blocks: Vec<AudioBlock>,

    // Output projection (optional — checkpoints without `a.output_proj.*` skip
    // this stage and feed the last conformer block straight into the projector).
    output_proj_w: Option<Vec<f32>>,
    output_proj_b: Option<Vec<f32>>,
    proj_fc_w: Vec<f32>,
    proj_fc_b: Option<Vec<f32>>,
    proj_input_w: Vec<f32>,
    proj_input_b: Option<Vec<f32>>,
}

impl AudioForward {
    /// Build the encoder. Reads every audio tensor through `WeightCache`'s
    /// async fetch path and dequantises into pure-Rust `Vec<f32>`s; the GPU
    /// is not touched at all on this path.
    pub async fn new(cfg: AudioConfig, wcache: Arc<WeightCache>) -> Result<Self> {
        let r = wcache.reader();

        // SSCP. The Conv2D weight shapes are stored in GGUF as [kW, kH, in_C, out_C]
        // with dim[0] fastest. Ollama's converter doesn't reshape these; we read
        // the raw bytes and reinterpret as f32.
        let sscp0_desc = r.tensor("a.conv1d.0.weight")?;
        let sscp0_w = dequant_tensor_to_f32_async(r, "a.conv1d.0.weight").await?;
        let sscp0_out_c = *sscp0_desc.dims.last().unwrap_or(&1) as usize;

        let sscp1_desc = r.tensor("a.conv1d.1.weight")?;
        let sscp1_w = dequant_tensor_to_f32_async(r, "a.conv1d.1.weight").await?;
        let sscp1_out_c = *sscp1_desc.dims.last().unwrap_or(&1) as usize;

        let sscp0_norm_w = dequant_tensor_to_f32_async(r, "a.conv1d.0.norm.weight").await?;
        let sscp0_norm_b = load_opt_f32(r, "a.conv1d.0.norm.bias").await?;
        let sscp1_norm_w = dequant_tensor_to_f32_async(r, "a.conv1d.1.norm.weight").await?;
        let sscp1_norm_b = load_opt_f32(r, "a.conv1d.1.norm.bias").await?;

        let pre_encode_out_w = dequant_tensor_to_f32_async(r, "a.pre_encode.out.weight").await?;
        let pre_encode_out_b = load_opt_f32(r, "a.pre_encode.out.bias").await?;

        // The pre_encode linear's input dim = (out_C1 * F'') where F'' is the
        // post-SSCP frequency dimension. Read it from the linear's k-axis (the
        // input axis is dims[0] in GGUF storage, i.e. the fast axis of the [k, n]
        // weight). Pre-encode weight shape: dim[0] = sscp_proj_in, dim[1] = hidden.
        let pre_desc = r.tensor("a.pre_encode.out.weight")?;
        let sscp_proj_in = *pre_desc.dims.first().unwrap_or(&1) as usize;

        // Per-block weights.
        let mut blocks = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            blocks.push(load_block(r, i).await?);
        }

        // Output projection + audio projector chain. The `a.output_proj.*`
        // tensors are absent from gemma4:e2b's GGUF — Ollama's `ForwardAudio`
        // skips this stage when nil, falling straight through to the projector.
        let output_proj_w = load_opt_f32(r, "a.output_proj.weight").await?;
        let output_proj_b = load_opt_f32(r, "a.output_proj.bias").await?;
        let proj_fc_w     = dequant_tensor_to_f32_async(r, "mm.a.fc.weight").await?;
        let proj_fc_b     = load_opt_f32(r, "mm.a.fc.bias").await?;
        let proj_input_w  = dequant_tensor_to_f32_async(r, "mm.a.input_projection.weight").await?;
        let proj_input_b  = load_opt_f32(r, "mm.a.input_projection.bias").await?;

        Ok(Self {
            cfg, mel: MelEngine::new(),
            sscp0_w, sscp0_norm_w, sscp0_norm_b,
            sscp1_w, sscp1_norm_w, sscp1_norm_b,
            pre_encode_out_w, pre_encode_out_b,
            sscp0_out_c, sscp1_out_c, sscp_proj_in,
            blocks,
            output_proj_w, output_proj_b,
            proj_fc_w, proj_fc_b,
            proj_input_w, proj_input_b,
        })
    }

    pub fn cfg(&self) -> &AudioConfig { &self.cfg }

    /// Compute a log-mel spectrogram from `samples` (16 kHz mono f32, [-1, 1]).
    /// Returns the flat `[n_frames * mel_bins]` tensor and the frame count.
    pub fn mel_spectrogram(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.mel.log_mel(samples)
    }

    /// Encode raw 16 kHz mono PCM into a flat `[n_audio_tokens * d_text]` slice
    /// of soft-token embeddings, ready for `Forward::step_with_embedding`.
    /// Mirrors `model_audio.go::AudioModel.ForwardAudio`. Pure CPU f32.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let mel_bins = cfg.mel_bins as usize;
        let eps = cfg.eps;
        let gc = cfg.grad_clip;

        // ---- 1. Log-mel spectrogram ----
        let (mel, n_frames) = self.mel.log_mel(samples);
        if n_frames == 0 {
            return Ok(Vec::new());
        }

        // ---- 2. SSCP: 2× Conv2D 3×3 stride=2 pad=1 + LayerNorm + ReLU ----
        // Treat mel as [1, n_frames, mel_bins] (single-channel image).
        let mut x = self.sscp_conv_block(
            &mel, 1, n_frames, mel_bins,
            self.sscp0_out_c,
            &self.sscp0_w, &self.sscp0_norm_w, self.sscp0_norm_b.as_deref(),
        );
        let t1 = (n_frames + 1) / 2;
        let f1 = (mel_bins + 1) / 2;
        x = self.sscp_conv_block(
            &x, self.sscp0_out_c, t1, f1,
            self.sscp1_out_c,
            &self.sscp1_w, &self.sscp1_norm_w, self.sscp1_norm_b.as_deref(),
        );
        let t_out = (t1 + 1) / 2;
        let f_out = (f1 + 1) / 2;
        // Layout coming out of sscp_conv_block: [t_out, f_out, channels] flat.
        // Reshape to [t_out, f_out * channels] for the linear projection.
        // (Ollama's permute (1,2,0,3) puts channels first then F''; we flatten.)
        // pre_encode.out.weight has dim[0] = sscp_proj_in. Detect what flat dim
        // matches and reshape accordingly.
        let flat_per_frame = f_out * self.sscp1_out_c;
        if flat_per_frame != self.sscp_proj_in {
            return Err(RullamaError::Inference(format!(
                "audio SSCP: flat per-frame dim {flat_per_frame} != pre_encode k {}",
                self.sscp_proj_in
            )));
        }

        // ---- 3. Linear projection to hidden ----
        let mut h = Self::linear_rows(
            &x, &self.pre_encode_out_w, self.pre_encode_out_b.as_deref(),
            t_out, self.sscp_proj_in, hidden,
        );
        let mut seq = t_out;

        // ---- 4. 12 Conformer blocks ----
        for b in 0..self.blocks.len() {
            self.forward_ffw(
                &mut h, seq,
                &self.blocks[b].ffw_norm,
                &self.blocks[b].ffw_up_w,   self.blocks[b].ffw_up_b.as_deref(),   &self.blocks[b].cl_ffw_up,
                &self.blocks[b].ffw_down_w, self.blocks[b].ffw_down_b.as_deref(), &self.blocks[b].cl_ffw_down,
                &self.blocks[b].ffw_post_norm,
            );
            self.forward_attention(&mut h, seq, &self.blocks[b]);
            self.forward_lightconv(&mut h, seq, &self.blocks[b]);
            self.forward_ffw(
                &mut h, seq,
                &self.blocks[b].ffw_norm_1,
                &self.blocks[b].ffw_up_1_w,   self.blocks[b].ffw_up_1_b.as_deref(),   &self.blocks[b].cl_ffw_up_1,
                &self.blocks[b].ffw_down_1_w, self.blocks[b].ffw_down_1_b.as_deref(), &self.blocks[b].cl_ffw_down_1,
                &self.blocks[b].ffw_post_norm_1,
            );
            // Final block: clamp + RMSNorm with the block-level pre_norm weight
            // (Ollama re-uses `Norm` as the BLOCK FINAL norm — see the closing
            // lines of AudioConformerBlock.Forward).
            for v in h.iter_mut() { *v = v.clamp(-gc, gc); }
            Self::rmsnorm_rows(&mut h, seq, hidden, Some(&self.blocks[b].pre_norm), eps);
        }

        // ---- 5. Optional output projection (skipped when not bundled in GGUF) ----
        let o = if let Some(opw) = self.output_proj_w.as_deref() {
            Self::linear_rows(&h, opw, self.output_proj_b.as_deref(), seq, hidden, hidden)
        } else {
            h
        };

        // ---- 6. Audio multimodal projector: FC + bias → unweighted RMSNorm
        //         → ClippableLinear input_projection ----
        let d_text = cfg.d_text as usize;
        let mut p = Self::linear_rows(
            &o, &self.proj_fc_w, self.proj_fc_b.as_deref(),
            seq, hidden, d_text,
        );
        Self::rmsnorm_rows(&mut p, seq, d_text, None, eps);
        // input_projection: d_text → d_text (square; clamp behaviour skipped — the
        // GGUF doesn't ship per-linear clamp scalars for the audio path the way
        // it does for vision, so AudioClippableLinear's `outMax != 0` check is
        // false in practice for gemma4:e2b).
        let q = Self::linear_rows(
            &p, &self.proj_input_w, self.proj_input_b.as_deref(),
            seq, d_text, d_text,
        );
        let _ = &mut seq;

        Ok(q)
    }

    /// One SSCP block: Conv2D (kernel=3, stride=2, padding=1) → LayerNorm → ReLU.
    /// Input layout: `[T, F, C_in]` channel-LAST flat.
    /// Output layout: `[T_out, F_out, C_out]` channel-LAST flat.
    fn sscp_conv_block(
        &self,
        x: &[f32], in_c: usize, in_t: usize, in_f: usize,
        out_c: usize,
        weight: &[f32], norm_w: &[f32], norm_b: Option<&[f32]>,
    ) -> Vec<f32> {
        // Conv2D kernel layout from GGUF: dims = [kW, kH, in_C, out_C]
        // (kW fastest); element at (oC, iC, kH, kW) = weight[((oC*in_c + iC)*3 + kH)*3 + kW].
        // Spatial: stride=(2,2), padding=(1,1), dilation=1.
        let k_h = 3usize;
        let k_w = 3usize;
        let s = 2usize;
        let pad = 1usize;
        let out_t = (in_t + 2 * pad).saturating_sub(k_h) / s + 1;
        let out_f = (in_f + 2 * pad).saturating_sub(k_w) / s + 1;
        let mut y = vec![0f32; out_t * out_f * out_c];

        for ot in 0..out_t {
            for of in 0..out_f {
                let in_t_base = (ot * s) as i64 - pad as i64;
                let in_f_base = (of * s) as i64 - pad as i64;
                for oc in 0..out_c {
                    let mut acc = 0f32;
                    for ic in 0..in_c {
                        for kh in 0..k_h {
                            let it = in_t_base + kh as i64;
                            if it < 0 || it >= in_t as i64 { continue; }
                            for kw in 0..k_w {
                                let if_ = in_f_base + kw as i64;
                                if if_ < 0 || if_ >= in_f as i64 { continue; }
                                let xi = ((it as usize) * in_f + if_ as usize) * in_c + ic;
                                let wi = ((oc * in_c + ic) * k_h + kh) * k_w + kw;
                                acc += x[xi] * weight[wi];
                            }
                        }
                    }
                    y[(ot * out_f + of) * out_c + oc] = acc;
                }
            }
        }

        // LayerNorm across the channel axis (per spatial position), then ReLU.
        for ot in 0..out_t {
            for of in 0..out_f {
                let off = (ot * out_f + of) * out_c;
                let row = &mut y[off..off + out_c];
                let mean: f32 = row.iter().sum::<f32>() / out_c as f32;
                let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out_c as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for c in 0..out_c {
                    let normed = (row[c] - mean) * inv * norm_w[c]
                        + norm_b.map(|b| b[c]).unwrap_or(0.0);
                    row[c] = normed.max(0.0); // ReLU
                }
            }
        }
        y
    }

    // ---- CPU primitives (per-frame ops; channel-LAST [seq, channels] layout) ----

    /// In-place RMSNorm with optional learned weight.
    /// `x` is `[seq, dim]`; norms each row of `dim` independently.
    pub fn rmsnorm_rows(x: &mut [f32], seq: usize, dim: usize, weight: Option<&[f32]>, eps: f32) {
        for r in 0..seq {
            let row = &mut x[r * dim..(r + 1) * dim];
            let mut sum_sq = 0f32;
            for &v in row.iter() { sum_sq += v * v; }
            let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            if let Some(w) = weight {
                for i in 0..dim { row[i] = row[i] * inv_rms * w[i]; }
            } else {
                for v in row.iter_mut() { *v = *v * inv_rms; }
            }
        }
    }

    /// `y[s, n] = Σ_k x[s, k] * w[n, k]` (+ optional `b[n]`).
    /// `w` shape `[k_dim, n_dim]` (GGUF dim[0] = k = fast axis).
    pub fn linear_rows(
        x: &[f32], w: &[f32], b: Option<&[f32]>,
        seq: usize, k_dim: usize, n_dim: usize,
    ) -> Vec<f32> {
        let mut y = vec![0f32; seq * n_dim];
        for s in 0..seq {
            for n in 0..n_dim {
                let mut acc = 0f32;
                for k in 0..k_dim {
                    acc += x[s * k_dim + k] * w[n * k_dim + k];
                }
                if let Some(bias) = b { acc += bias[n]; }
                y[s * n_dim + n] = acc;
            }
        }
        y
    }

    /// `linear_rows` with the `AudioClippableLinear` semantics around it:
    /// optional input clamp before the matmul, optional output clamp after.
    /// Each clamp activates iff its max scalar is non-zero (matches Ollama's
    /// `if l.inMax != 0 { ... }`).
    fn clipped_linear_rows(
        x: &[f32], w: &[f32], b: Option<&[f32]>,
        rows: usize, in_dim: usize, out_dim: usize,
        clamp: &Clamp,
    ) -> Vec<f32> {
        let mut y = if clamp.in_max != 0.0 {
            let mut xc = x.to_vec();
            clamp.apply_in(&mut xc);
            Self::linear_rows(&xc, w, b, rows, in_dim, out_dim)
        } else {
            Self::linear_rows(x, w, b, rows, in_dim, out_dim)
        };
        clamp.apply_out(&mut y);
        y
    }

    /// FFW with half-residual: `x = residual + 0.5 * (x → clamp → norm → up
    /// → SiLU → down → clamp → post_norm)`. In-place.
    fn forward_ffw(
        &self,
        x: &mut [f32],
        seq: usize,
        norm_w: &[f32],
        up_w: &[f32], up_b: Option<&[f32]>, up_clamp: &Clamp,
        down_w: &[f32], down_b: Option<&[f32]>, down_clamp: &Clamp,
        post_norm_w: &[f32],
    ) {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let ffn = cfg.ffn_inter as usize;
        let eps = cfg.eps;
        let gc = cfg.grad_clip;
        let rw = cfg.residual_w;

        let residual: Vec<f32> = x.to_vec();
        for v in x.iter_mut() { *v = v.clamp(-gc, gc); }
        Self::rmsnorm_rows(x, seq, hidden, Some(norm_w), eps);
        // Up linear (clipped).
        let mut h = Self::clipped_linear_rows(x, up_w, up_b, seq, hidden, ffn, up_clamp);
        // SiLU.
        for v in h.iter_mut() { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        // Down linear (clipped).
        let mut h = Self::clipped_linear_rows(&h, down_w, down_b, seq, ffn, hidden, down_clamp);
        for v in h.iter_mut() { *v = v.clamp(-gc, gc); }
        Self::rmsnorm_rows(&mut h, seq, hidden, Some(post_norm_w), eps);
        for i in 0..x.len() {
            x[i] = residual[i] + rw * h[i];
        }
    }

    /// Block-local self-attention with relative-position bias and softcap.
    /// Mirrors `model_audio.go::AudioConformerBlock.forwardAttention`.
    ///
    /// Runs per chunk of `chunk_size` queries against a `context_size`-wide
    /// key/value window (max_past zeros on left + max_future+chunk_size-1 on right).
    /// Each (query, context) score has:
    ///   * Content-content: scaled q · k
    ///   * Content-position: q · projPos[p] where p = max_past + r - c
    ///   * Causal-valid mask
    ///   * Logit softcap = `cfg.logit_cap` (50.0)
    ///   * Softmax over context dimension
    fn forward_attention(&self, x: &mut [f32], seq: usize, block: &AudioBlock) {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let n_heads = cfg.n_heads as usize;
        let head_dim = cfg.head_dim() as usize;
        let chunk_size = cfg.chunk_size as usize;
        let max_past = cfg.max_past as usize;
        let max_future = cfg.max_future as usize;
        let context_size = cfg.context_size as usize;
        let max_span = max_past + max_future + 1;
        let cap = cfg.logit_cap;
        let eps = cfg.eps;
        let gc = cfg.grad_clip;

        let residual: Vec<f32> = x.to_vec();
        for v in x.iter_mut() { *v = v.clamp(-gc, gc); }
        Self::rmsnorm_rows(x, seq, hidden, Some(&block.attn_pre_norm), eps);

        // Q, K, V projections (each is a ClippableLinear).
        let mut q = Self::clipped_linear_rows(x, &block.attn_q_w, block.attn_q_b.as_deref(),
            seq, hidden, hidden, &block.cl_attn_q);
        let mut k = Self::clipped_linear_rows(x, &block.attn_k_w, block.attn_k_b.as_deref(),
            seq, hidden, hidden, &block.cl_attn_k);
        let v = Self::clipped_linear_rows(x, &block.attn_v_w, block.attn_v_b.as_deref(),
            seq, hidden, hidden, &block.cl_attn_v);

        // Per-dim Q scale: (head_dim^-0.5 / ln 2) * per_dim_scale (already softplus'd
        // by Ollama's converter — model_audio.go::forwardAttention line 305-309).
        // per_dim_scale is shared across heads; broadcast over the head axis.
        let q_scale_base = (head_dim as f32).powf(-0.5) / std::f32::consts::LN_2;
        for s in 0..seq {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    q[s * hidden + h * head_dim + d] *= q_scale_base * block.per_dim_scale[d];
                }
            }
        }
        // K scale: softplus(1) / ln 2 = ln(1 + e) / ln 2 ≈ 1.886.
        let k_scale = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        for kv in k.iter_mut() { *kv *= k_scale; }

        // Sinusoidal position embeddings → projection through linear_pos.
        // Same layout as q/k: [max_span, n_heads * head_dim] flat.
        let half_dim = hidden / 2;
        let mut pos_emb = vec![0f32; max_span * hidden];
        let log_inc = (10000f32).ln() / (half_dim.saturating_sub(1)).max(1) as f32;
        for p in 0..max_span {
            let rel_pos = (max_past as f32) - (p as f32);
            for d in 0..half_dim {
                let angle = rel_pos * (-(d as f32) * log_inc).exp();
                pos_emb[p * hidden + d] = angle.sin();
                pos_emb[p * hidden + half_dim + d] = angle.cos();
            }
        }
        // Project: [max_span, hidden] × [hidden, hidden] (linear_pos) → [max_span, hidden].
        let pos_proj = Self::linear_rows(&pos_emb, &block.linear_pos, None,
            max_span, hidden, hidden);

        // Pad q/k/v on the right so seq divides chunk_size.
        let num_chunks = (seq + chunk_size - 1) / chunk_size;
        let padded_len = num_chunks * chunk_size;
        let mut q_pad = q;
        q_pad.resize(padded_len * hidden, 0.0);
        let mut k_inner = k;
        k_inner.resize(padded_len * hidden, 0.0);
        let mut v_inner = v;
        v_inner.resize(padded_len * hidden, 0.0);

        // Pad k/v: max_past zeros on left, max_future + chunk_size - 1 on right.
        let pad_left = max_past;
        let pad_right = max_future + chunk_size - 1;
        let k_padded_len = pad_left + padded_len + pad_right;
        let mut k_padded = vec![0f32; k_padded_len * hidden];
        let mut v_padded = vec![0f32; k_padded_len * hidden];
        k_padded[pad_left * hidden..(pad_left + padded_len) * hidden]
            .copy_from_slice(&k_inner);
        v_padded[pad_left * hidden..(pad_left + padded_len) * hidden]
            .copy_from_slice(&v_inner);

        let mut attn_out = vec![0f32; padded_len * hidden];

        for u in 0..num_chunks {
            for r in 0..chunk_size {
                for h in 0..n_heads {
                    let q_off = (u * chunk_size + r) * hidden + h * head_dim;

                    // Compute logits per context position; track max for stable softmax.
                    let mut logits = vec![f32::NEG_INFINITY; context_size];
                    let mut max_logit = f32::NEG_INFINITY;

                    for c in 0..context_size {
                        // Causal-valid mask.
                        let actual_t = (u * chunk_size) as i64 + c as i64 - pad_left as i64;
                        let valid = actual_t >= 0 && actual_t < seq as i64;
                        let causal = c >= r && c <= r + max_past + max_future;
                        if !valid || !causal { continue; }

                        // Content-content score.
                        let k_off = (u * chunk_size + c) * hidden + h * head_dim;
                        let mut ac = 0f32;
                        for d in 0..head_dim {
                            ac += q_pad[q_off + d] * k_padded[k_off + d];
                        }

                        // Content-position score: lookup pos_proj at p = max_past + r - c.
                        let p_signed = max_past as i64 + r as i64 - c as i64;
                        let bd = if p_signed >= 0 && (p_signed as usize) < max_span {
                            let p = p_signed as usize;
                            let pos_off = p * hidden + h * head_dim;
                            let mut bd = 0f32;
                            for d in 0..head_dim {
                                bd += q_pad[q_off + d] * pos_proj[pos_off + d];
                            }
                            bd
                        } else { 0.0 };

                        let mut score = ac + bd;
                        // Logit softcap: tanh(score / cap) * cap.
                        score = (score / cap).tanh() * cap;
                        logits[c] = score;
                        if score > max_logit { max_logit = score; }
                    }

                    // Softmax over the context dim.
                    let mut sum_exp = 0f32;
                    for c in 0..context_size {
                        if logits[c] == f32::NEG_INFINITY {
                            logits[c] = 0.0;
                            continue;
                        }
                        let e = (logits[c] - max_logit).exp();
                        logits[c] = e;
                        sum_exp += e;
                    }
                    let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

                    // Weighted V sum into attn_out[(u*chunk_size + r), h, :].
                    let out_off = (u * chunk_size + r) * hidden + h * head_dim;
                    for d in 0..head_dim {
                        let mut acc = 0f32;
                        for c in 0..context_size {
                            if logits[c] == 0.0 { continue; }
                            let weight = logits[c] * inv;
                            let v_off = (u * chunk_size + c) * hidden + h * head_dim;
                            acc += weight * v_padded[v_off + d];
                        }
                        attn_out[out_off + d] = acc;
                    }
                }
            }
        }

        // Trim back to seq.
        attn_out.truncate(seq * hidden);

        // Output projection (ClippableLinear) + grad clamp + post-norm + residual.
        let mut o = Self::clipped_linear_rows(&attn_out, &block.attn_o_w, block.attn_o_b.as_deref(),
            seq, hidden, hidden, &block.cl_attn_o);
        for v in o.iter_mut() { *v = v.clamp(-gc, gc); }
        Self::rmsnorm_rows(&mut o, seq, hidden, Some(&block.attn_post_norm), eps);
        for i in 0..x.len() {
            x[i] = residual[i] + o[i];
        }
    }

    /// LightConv: `x = residual + (x → norm → pw1 → GLU → depthwise → clamp →
    /// norm_conv → SiLU → pw2)`. In-place into `x`.
    fn forward_lightconv(
        &self,
        x: &mut [f32],
        seq: usize,
        block: &AudioBlock,
    ) {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let kernel = cfg.conv_kernel as usize;
        let eps = cfg.eps;
        let gc = cfg.grad_clip;

        // Save residual.
        let residual: Vec<f32> = x.to_vec();
        // norm.
        Self::rmsnorm_rows(x, seq, hidden, Some(&block.conv_norm), eps);
        // conv_pw1 (ClippableLinear): hidden -> 2*hidden
        let h = Self::clipped_linear_rows(x, &block.conv_pw1_w, block.conv_pw1_b.as_deref(),
            seq, hidden, hidden * 2, &block.cl_conv_pw1);
        // GLU split: data half * sigmoid(gate half)
        let mut g = vec![0f32; seq * hidden];
        for s in 0..seq {
            for d in 0..hidden {
                let data = h[s * hidden * 2 + d];
                let gate = h[s * hidden * 2 + hidden + d];
                let sig = 1.0 / (1.0 + (-gate).exp());
                g[s * hidden + d] = data * sig;
            }
        }
        // Depthwise conv1d (kernel=5, left zero-pad). conv_dw is [hidden, kernel].
        let mut conv_out = vec![0f32; seq * hidden];
        for t in 0..seq {
            for c in 0..hidden {
                let mut acc = 0f32;
                for k in 0..kernel {
                    let shift = kernel - 1 - k;
                    if t < shift { continue; }
                    let src_t = t - shift;
                    acc += g[src_t * hidden + c] * block.conv_dw[c * kernel + k];
                }
                conv_out[t * hidden + c] = acc;
            }
        }
        // Clamp + norm_conv + SiLU.
        for v in conv_out.iter_mut() { *v = v.clamp(-gc, gc); }
        Self::rmsnorm_rows(&mut conv_out, seq, hidden, Some(&block.norm_conv), eps);
        for v in conv_out.iter_mut() { *v = *v * (1.0 / (1.0 + (-*v).exp())); }
        // conv_pw2 (ClippableLinear).
        let pw2_out = Self::clipped_linear_rows(&conv_out, &block.conv_pw2_w,
            block.conv_pw2_b.as_deref(), seq, hidden, hidden, &block.cl_conv_pw2);
        // Residual.
        for i in 0..x.len() {
            x[i] = residual[i] + pw2_out[i];
        }
    }
}

async fn load_opt_f32(r: &GgufReader, name: &str) -> Result<Option<Vec<f32>>> {
    match r.tensor(name) {
        Ok(_) => Ok(Some(dequant_tensor_to_f32_async(r, name).await?)),
        Err(_) => Ok(None),
    }
}

async fn load_block(r: &GgufReader, i: u32) -> Result<AudioBlock> {
    let p = format!("a.blk.{i}.");
    let load = |suffix: &str| -> Result<String> { Ok(format!("{p}{suffix}")) };
    Ok(AudioBlock {
        pre_norm:        dequant_tensor_to_f32_async(r, &load("layer_pre_norm.weight")?).await?,
        ffw_norm:        dequant_tensor_to_f32_async(r, &load("ffn_norm.weight")?).await?,
        ffw_up_w:        dequant_tensor_to_f32_async(r, &load("ffn_up.weight")?).await?,
        ffw_up_b:        load_opt_f32(r, &load("ffn_up.bias")?).await?,
        ffw_down_w:      dequant_tensor_to_f32_async(r, &load("ffn_down.weight")?).await?,
        ffw_down_b:      load_opt_f32(r, &load("ffn_down.bias")?).await?,
        ffw_post_norm:   dequant_tensor_to_f32_async(r, &load("ffn_post_norm.weight")?).await?,
        ffw_norm_1:      dequant_tensor_to_f32_async(r, &load("ffn_norm_1.weight")?).await?,
        ffw_up_1_w:      dequant_tensor_to_f32_async(r, &load("ffn_up_1.weight")?).await?,
        ffw_up_1_b:      load_opt_f32(r, &load("ffn_up_1.bias")?).await?,
        ffw_down_1_w:    dequant_tensor_to_f32_async(r, &load("ffn_down_1.weight")?).await?,
        ffw_down_1_b:    load_opt_f32(r, &load("ffn_down_1.bias")?).await?,
        ffw_post_norm_1: dequant_tensor_to_f32_async(r, &load("ffn_post_norm_1.weight")?).await?,
        attn_pre_norm:   dequant_tensor_to_f32_async(r, &load("ln1.weight")?).await?,
        attn_post_norm:  dequant_tensor_to_f32_async(r, &load("ln2.weight")?).await?,
        attn_q_w:        dequant_tensor_to_f32_async(r, &load("attn_q.weight")?).await?,
        attn_q_b:        load_opt_f32(r, &load("attn_q.bias")?).await?,
        attn_k_w:        dequant_tensor_to_f32_async(r, &load("attn_k.weight")?).await?,
        attn_k_b:        load_opt_f32(r, &load("attn_k.bias")?).await?,
        attn_v_w:        dequant_tensor_to_f32_async(r, &load("attn_v.weight")?).await?,
        attn_v_b:        load_opt_f32(r, &load("attn_v.bias")?).await?,
        attn_o_w:        dequant_tensor_to_f32_async(r, &load("attn_out.weight")?).await?,
        attn_o_b:        load_opt_f32(r, &load("attn_out.bias")?).await?,
        linear_pos:      dequant_tensor_to_f32_async(r, &load("linear_pos.weight")?).await?,
        per_dim_scale:   dequant_tensor_to_f32_async(r, &load("per_dim_scale.weight")?).await?,
        conv_norm:       dequant_tensor_to_f32_async(r, &load("conv_norm.weight")?).await?,
        norm_conv:       dequant_tensor_to_f32_async(r, &load("norm_conv.weight")?).await?,
        conv_pw1_w:      dequant_tensor_to_f32_async(r, &load("conv_pw1.weight")?).await?,
        conv_pw1_b:      load_opt_f32(r, &load("conv_pw1.bias")?).await?,
        conv_pw2_w:      dequant_tensor_to_f32_async(r, &load("conv_pw2.weight")?).await?,
        conv_pw2_b:      load_opt_f32(r, &load("conv_pw2.bias")?).await?,
        conv_dw:         dequant_tensor_to_f32_async(r, &load("conv_dw.weight")?).await?,

        cl_attn_q:    load_clamp(r, &format!("{p}attn_q")).await?,
        cl_attn_k:    load_clamp(r, &format!("{p}attn_k")).await?,
        cl_attn_v:    load_clamp(r, &format!("{p}attn_v")).await?,
        cl_attn_o:    load_clamp(r, &format!("{p}attn_out")).await?,
        cl_ffw_up:    load_clamp(r, &format!("{p}ffn_up")).await?,
        cl_ffw_down:  load_clamp(r, &format!("{p}ffn_down")).await?,
        cl_ffw_up_1:  load_clamp(r, &format!("{p}ffn_up_1")).await?,
        cl_ffw_down_1:load_clamp(r, &format!("{p}ffn_down_1")).await?,
        cl_conv_pw1:  load_clamp(r, &format!("{p}conv_pw1")).await?,
        cl_conv_pw2:  load_clamp(r, &format!("{p}conv_pw2")).await?,
    })
}
