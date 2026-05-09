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
//!     → output_proj (linear with bias)
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

    // Output projection + projector.
    output_proj_w: Vec<f32>,
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

        // Output projection + audio projector chain.
        let output_proj_w = dequant_tensor_to_f32_async(r, "a.output_proj.weight").await?;
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
    /// This is the only piece currently wired end-to-end; the Conformer body
    /// is the next chunk of work (see `project_m13_status.md`).
    pub fn mel_spectrogram(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.mel.log_mel(samples)
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
    })
}
