//! rullama-core — C-ABI FFI shim over the published `rullama` crate.
//!
//! ## The dominant constraint: the native `Model` is `!Send`
//!
//! `rullama`'s `TensorFetcher` is `#[async_trait(?Send)]` and the `Model` +
//! wgpu futures are single-threaded. A `Model` therefore cannot be moved
//! between threads. The pattern:
//!
//! * Each engine handle (`RlModel`) owns **one** OS thread for its entire
//!   lifetime. The `Model` (and `WgpuCtx`) are created on that thread and
//!   never leave it.
//! * Each C-ABI call marshals a [`Command`] to that thread over an MPSC
//!   channel and blocks for the reply. The exception is **cancellation**:
//!   while a generation runs the worker is busy in its decode loop and cannot
//!   service the channel, so `rl_cancel` flips a shared `AtomicBool` that the
//!   loop polls each step.
//! * Async engine methods are driven with `pollster::block_on` on the owning
//!   thread (the approach proven in upstream `tools/ios-bench`).
//!
//! Only `Send` values cross the channel; the `!Send` engine state stays put.
//! Streaming tokens are pushed through a C callback invoked on the worker
//! thread — the C# side hops them to the UI thread.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;

use rullama_engine::api::{Model, RomeIterativeHparams};
use rullama_engine::backend::WgpuCtx;
use rullama_engine::embed::EmbeddingModel;
use rullama_engine::gguf::{FileFetcher, TensorFetcher};
use rullama_engine::imagegen::{FileBlobSource, ImageBundle, VaeConfig, rgb_chw_to_rgba8};
use rullama_engine::sampling::SamplingOptions;
use rullama_engine::styletts2_clone::StyleTtsClone;
use rullama_engine::tts::KokoroTts;
use rullama_lora::session::TrainingSession;
use rullama_lora::shared::config::{LoraConfig, LrScheduler, TrainingHyperparams};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Thread-local last-error (set on, and read from, the calling thread)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let c = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Last error message for the **calling thread** as a NUL-terminated C string,
/// or NULL. Valid until the next FFI call on this thread; copy it immediately.
#[unsafe(no_mangle)]
pub extern "C" fn rl_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
}

/// This shim's version string (static; do not free).
#[unsafe(no_mangle)]
pub extern "C" fn rl_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Frees a C string previously returned by this library.
///
/// # Safety
/// `ptr` must be NULL or a pointer obtained from this library, unused after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_free_str(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

/// Frees a `u32` array previously returned by this library (e.g. `rl_encode`).
///
/// # Safety
/// `ptr`/`n` must come from a single returned allocation, unused after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_free_u32(ptr: *mut u32, n: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n)) });
    }
}

fn into_c_string(s: impl Into<Vec<u8>>) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("<nul>").unwrap())
        .into_raw()
}

unsafe fn c_str<'a>(p: *const c_char) -> Result<&'a str, &'static str> {
    if p.is_null() {
        return Err("null string");
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| "invalid utf-8")
}

// ---------------------------------------------------------------------------
// Streaming callback (raw C function pointer + opaque context)
// ---------------------------------------------------------------------------

/// Per-token callback: `(ctx, token_id, piece, is_eos)`. `piece` is the decoded
/// display text (SentencePiece ▁ already mapped to space), valid only for the
/// duration of the call. Invoked on the worker thread.
type TokenFn = extern "C" fn(ctx: *mut c_void, token_id: u32, piece: *const c_char, is_eos: i32);

/// `Send` wrapper so the callback + ctx can cross the command channel. The C#
/// caller keeps `ctx` alive for the duration of the `rl_generate` call.
struct TokenCb {
    f: TokenFn,
    ctx: *mut c_void,
}
unsafe impl Send for TokenCb {}

// ---------------------------------------------------------------------------
// Owning-thread command loop
// ---------------------------------------------------------------------------

enum Command {
    Probe(Sender<Result<String, String>>),
    LoadPath {
        path: String,
        max_ctx: u32,
        text_only: bool,
        reply: Sender<Result<(), String>>,
    },
    Encode {
        text: String,
        reply: Sender<Result<Vec<u32>, String>>,
    },
    TokenStr {
        id: u32,
        reply: Sender<Option<String>>,
    },
    SetSampling {
        opts: SamplingOptions,
        reply: Sender<()>,
    },
    Reset(Sender<()>),
    VocabSize(Sender<u32>),
    Position(Sender<u32>),
    Generate {
        prompt: Vec<u32>,
        max_new: u32,
        cb: TokenCb,
        reply: Sender<Result<u32, String>>,
    },
    HasVision(Sender<bool>),
    HasAudio(Sender<bool>),
    Sentinels {
        audio: bool,
        reply: Sender<Option<(u32, u32)>>,
    },
    ImageSoftCount {
        h: usize,
        w: usize,
        reply: Sender<Option<usize>>,
    },
    EncodeImage {
        pixels: Vec<f32>,
        h: usize,
        w: usize,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    EncodeAudio {
        pcm: Vec<f32>,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    /// Generate with a single spliced media item: during prefill, after the
    /// `sentinel_begin` token is fed, the soft-token rows in `soft`
    /// (`soft.len()/d_text` rows) are spliced via step_with_embedding.
    GenerateSpliced {
        prompt: Vec<u32>,
        sentinel_begin: u32,
        soft: Vec<f32>,
        d_text: usize,
        max_new: u32,
        cb: TokenCb,
        reply: Sender<Result<u32, String>>,
    },
    /// Convert the loaded Model into a LoRA TrainingSession (consumes the Model).
    TrainerBegin {
        rank: u32,
        alpha: f32,
        dropout: f32,
        target_modules: Vec<String>,
        max_seq_len: usize,
        learning_rate: f64,
        reply: Sender<Result<(), String>>,
    },
    /// One training step on (input_ids → target); returns the loss.
    TrainerStep {
        input_ids: Vec<u32>,
        target: u32,
        reply: Sender<Result<f32, String>>,
    },
    /// Serialize the trained LoRA adapter to safetensors bytes.
    TrainerSaveAdapter {
        reply: Sender<Result<Vec<u8>, String>>,
    },
    /// Load a LoRA adapter (safetensors bytes) into the chat model.
    LoadAdapter {
        bytes: Vec<u8>,
        reply: Sender<Result<usize, String>>,
    },
    ClearAdapter(Sender<()>),
    /// ROME knowledge edit: make `subject` (in `prompt`) predict `target` at `layer`.
    RomeEdit {
        prompt: String,
        subject: String,
        target: String,
        layer: u32,
        reply: Sender<Result<(), String>>,
    },
    Shutdown,
}

fn worker(rx: mpsc::Receiver<Command>, cancel: Arc<AtomicBool>) {
    // `!Send` engine state — created and dropped only on this thread.
    let mut ctx: Option<WgpuCtx> = None;
    let mut model: Option<Model> = None;
    let mut trainer: Option<TrainingSession> = None;

    for cmd in rx {
        match cmd {
            Command::Probe(reply) => {
                if ctx.is_none() {
                    match pollster::block_on(WgpuCtx::new()) {
                        Ok(c) => ctx = Some(c),
                        Err(e) => {
                            let _ = reply.send(Err(format!("wgpu init failed: {e:?}")));
                            continue;
                        }
                    }
                }
                let c = ctx.as_ref().expect("ctx set above");
                let info = c.adapter.get_info();
                let limits = c.adapter.limits();
                let _ = reply.send(Ok(format!(
                    "adapter={} backend={:?} device_type={:?} subgroups={} has_f16={} max_storage_buffer_binding_size={}",
                    info.name, info.backend, info.device_type, c.has_subgroups, c.has_f16,
                    limits.max_storage_buffer_binding_size,
                )));
            }
            Command::LoadPath {
                path,
                max_ctx,
                text_only,
                reply,
            } => {
                let res = (|| -> Result<Model, String> {
                    let fetcher = FileFetcher::open(std::path::Path::new(&path))
                        .map_err(|e| format!("{e}"))?;
                    let arc: Arc<dyn TensorFetcher> = Arc::new(fetcher);
                    let m = if text_only {
                        pollster::block_on(Model::load_streaming_text_only(arc, max_ctx))
                    } else {
                        pollster::block_on(Model::load_streaming_with_max_context(arc, max_ctx))
                    }
                    .map_err(|e| format!("{e}"))?;
                    Ok(m)
                })();
                match res {
                    Ok(m) => {
                        model = Some(m);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            Command::Encode { text, reply } => {
                // Fall back to the trainer's model so tokenization works during training.
                let mref = model
                    .as_ref()
                    .or_else(|| trainer.as_ref().map(|t| t.model()));
                let _ = reply.send(match mref {
                    Some(m) => Ok(m.encode_tokens(&text)),
                    None => Err("no model loaded".into()),
                });
            }
            Command::TokenStr { id, reply } => {
                let mref = model
                    .as_ref()
                    .or_else(|| trainer.as_ref().map(|t| t.model()));
                let _ = reply.send(mref.and_then(|m| m.token_str_native(id)));
            }
            Command::SetSampling { opts, reply } => {
                if let Some(m) = model.as_mut() {
                    m.set_sampling_native(opts);
                }
                let _ = reply.send(());
            }
            Command::Reset(reply) => {
                if let Some(m) = model.as_mut() {
                    m.reset_native();
                }
                let _ = reply.send(());
            }
            Command::VocabSize(reply) => {
                let mref = model
                    .as_ref()
                    .or_else(|| trainer.as_ref().map(|t| t.model()));
                let _ = reply.send(mref.map_or(0, |m| m.vocab_size_native()));
            }
            Command::Position(reply) => {
                let _ = reply.send(model.as_ref().map_or(0, |m| m.position_native()));
            }
            Command::Generate {
                prompt,
                max_new,
                cb,
                reply,
            } => {
                let Some(m) = model.as_mut() else {
                    let _ = reply.send(Err("no model loaded".into()));
                    continue;
                };
                cancel.store(false, Ordering::SeqCst);
                let res = (|| -> Result<u32, String> {
                    if prompt.is_empty() {
                        return Err("empty prompt".into());
                    }
                    // Prefill: feed every prompt token; the last step yields the
                    // first sampled continuation token.
                    let mut cur = 0u32;
                    for &tok in &prompt {
                        cur = pollster::block_on(m.step_native(tok)).map_err(|e| format!("{e}"))?;
                    }
                    decode_loop(m, cur, max_new, &cb, &cancel)
                })();
                let _ = reply.send(res);
            }
            Command::HasVision(reply) => {
                let _ = reply.send(model.as_ref().is_some_and(|m| m.has_vision_native()));
            }
            Command::HasAudio(reply) => {
                let _ = reply.send(model.as_ref().is_some_and(|m| m.has_audio_native()));
            }
            Command::Sentinels { audio, reply } => {
                let r = model.as_ref().and_then(|m| {
                    if audio {
                        m.audio_sentinel_ids_native()
                    } else {
                        m.image_sentinel_ids_native()
                    }
                });
                let _ = reply.send(r);
            }
            Command::ImageSoftCount { h, w, reply } => {
                let _ = reply.send(
                    model
                        .as_ref()
                        .and_then(|m| m.image_soft_token_count_native(h, w)),
                );
            }
            Command::EncodeImage {
                pixels,
                h,
                w,
                reply,
            } => {
                let res = match model.as_mut() {
                    Some(m) => pollster::block_on(m.encode_image_native(&pixels, h, w, None))
                        .map_err(|e| format!("{e}")),
                    None => Err("no model loaded".into()),
                };
                let _ = reply.send(res);
            }
            Command::EncodeAudio { pcm, reply } => {
                let res = match model.as_mut() {
                    Some(m) => {
                        pollster::block_on(m.encode_audio_native(&pcm)).map_err(|e| format!("{e}"))
                    }
                    None => Err("no model loaded".into()),
                };
                let _ = reply.send(res);
            }
            Command::GenerateSpliced {
                prompt,
                sentinel_begin,
                soft,
                d_text,
                max_new,
                cb,
                reply,
            } => {
                let Some(m) = model.as_mut() else {
                    let _ = reply.send(Err("no model loaded".into()));
                    continue;
                };
                cancel.store(false, Ordering::SeqCst);
                let res = (|| -> Result<u32, String> {
                    if prompt.is_empty() {
                        return Err("empty prompt".into());
                    }
                    if d_text == 0 || soft.len() % d_text != 0 {
                        return Err("bad soft-token dims".into());
                    }
                    let n_soft = soft.len() / d_text;
                    // Prefill with splice at the begin sentinel.
                    let mut cur = 0u32;
                    for &id in &prompt {
                        cur = pollster::block_on(m.step_native(id)).map_err(|e| format!("{e}"))?;
                        if id == sentinel_begin {
                            for r in 0..n_soft {
                                let row = &soft[r * d_text..(r + 1) * d_text];
                                cur = pollster::block_on(m.step_with_embedding_native(row))
                                    .map_err(|e| format!("{e}"))?;
                            }
                        }
                    }
                    decode_loop(m, cur, max_new, &cb, &cancel)
                })();
                let _ = reply.send(res);
            }
            Command::TrainerBegin {
                rank,
                alpha,
                dropout,
                target_modules,
                max_seq_len,
                learning_rate,
                reply,
            } => {
                let Some(m) = model.take() else {
                    let _ = reply.send(Err("no model loaded".into()));
                    continue;
                };
                let lora = LoraConfig {
                    rank,
                    alpha,
                    dropout,
                    target_modules: if target_modules.is_empty() {
                        LoraConfig::default().target_modules
                    } else {
                        target_modules
                    },
                    ..Default::default()
                };
                let hp = TrainingHyperparams {
                    epochs: 1,
                    batch_size: 1,
                    gradient_accumulation_steps: 1,
                    warmup_steps: 0,
                    lr_scheduler: LrScheduler::Constant,
                    max_seq_len,
                    learning_rate,
                    ..Default::default()
                };
                match TrainingSession::new(m, lora, hp) {
                    Ok(t) => {
                        trainer = Some(t);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(format!("{e:?}")));
                    } // model consumed on error
                }
            }
            Command::TrainerStep {
                input_ids,
                target,
                reply,
            } => {
                let r = match trainer.as_mut() {
                    Some(t) => {
                        pollster::block_on(t.step(&input_ids, target)).map_err(|e| format!("{e:?}"))
                    }
                    None => Err("no trainer (call rl_trainer_begin)".into()),
                };
                let _ = reply.send(r);
            }
            Command::TrainerSaveAdapter { reply } => {
                let r = match trainer.as_ref() {
                    Some(t) => {
                        pollster::block_on(t.save_adapter_to_bytes()).map_err(|e| format!("{e:?}"))
                    }
                    None => Err("no trainer".into()),
                };
                let _ = reply.send(r);
            }
            Command::LoadAdapter { bytes, reply } => {
                let r = match model.as_mut() {
                    Some(m) => m.load_adapter_native(&bytes).map_err(|e| format!("{e}")),
                    None => Err("no model loaded".into()),
                };
                let _ = reply.send(r);
            }
            Command::ClearAdapter(reply) => {
                if let Some(m) = model.as_mut() {
                    m.clear_adapter_native();
                }
                let _ = reply.send(());
            }
            Command::RomeEdit {
                prompt,
                subject,
                target,
                layer,
                reply,
            } => {
                let r = (|| -> Result<(), String> {
                    let m = model.as_mut().ok_or("no model loaded")?;
                    let prompt_ids = m.encode_tokens(&prompt);
                    let t = if target.starts_with(' ') {
                        target.clone()
                    } else {
                        format!(" {target}")
                    };
                    let target_ids = m.encode_tokens(&t);
                    let target_id = *target_ids.first().ok_or("empty target")?;
                    let pos = m
                        .find_subject_last_pos(&prompt_ids, &subject)
                        .ok_or("subject not found in prompt")?;
                    pollster::block_on(m.rome_edit_iterative_native(
                        &prompt_ids,
                        pos,
                        layer,
                        target_id,
                        RomeIterativeHparams::default(),
                    ))
                    .map_err(|e| format!("{e}"))?;
                    Ok(())
                })();
                let _ = reply.send(r);
            }
            Command::Shutdown => break,
        }
    }
}

/// Shared decode loop: emit decoded pieces until EOS / max_new / cancel.
fn decode_loop(
    m: &mut Model,
    mut cur: u32,
    max_new: u32,
    cb: &TokenCb,
    cancel: &AtomicBool,
) -> Result<u32, String> {
    let mut produced = 0u32;
    while produced < max_new {
        if cancel.load(Ordering::SeqCst) || m.is_eos_native(cur) {
            break;
        }
        let piece = m
            .token_str_native(cur)
            .unwrap_or_default()
            .replace('\u{2581}', " ");
        let cpiece = CString::new(piece).unwrap_or_default();
        (cb.f)(cb.ctx, cur, cpiece.as_ptr(), 0);
        produced += 1;
        cur = pollster::block_on(m.step_native(cur)).map_err(|e| format!("{e}"))?;
    }
    Ok(produced)
}

/// Opaque engine handle: owns the worker thread, the channel, and the cancel flag.
pub struct RlModel {
    tx: Sender<Command>,
    handle: Option<JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
}

impl RlModel {
    fn from_ptr<'a>(m: *mut RlModel) -> Option<&'a RlModel> {
        unsafe { m.as_ref() }
    }
}

/// Creates an engine handle: spawns its dedicated owning thread.
#[unsafe(no_mangle)]
pub extern "C" fn rl_model_create() -> *mut RlModel {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = cancel.clone();
    match std::thread::Builder::new()
        .name("rullama-model".into())
        .spawn(move || worker(rx, cancel_worker))
    {
        Ok(handle) => Box::into_raw(Box::new(RlModel {
            tx,
            handle: Some(handle),
            cancel,
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn model thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Frees an engine handle: stops + joins the owning thread (dropping the
/// `!Send` engine state on its own thread).
///
/// # Safety
/// `m` must be NULL or a handle from `rl_model_create`, unused afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_model_free(m: *mut RlModel) {
    if m.is_null() {
        return;
    }
    let mut model = unsafe { Box::from_raw(m) };
    model.cancel.store(true, Ordering::SeqCst);
    let _ = model.tx.send(Command::Shutdown);
    if let Some(h) = model.handle.take() {
        let _ = h.join();
    }
}

// Small helper: send a command and block for its reply.
fn call<T>(m: *mut RlModel, make: impl FnOnce(Sender<T>) -> Command) -> Result<T, i32> {
    let Some(model) = RlModel::from_ptr(m) else {
        set_last_error("null handle");
        return Err(-1);
    };
    let (tx, rx) = mpsc::channel();
    if model.tx.send(make(tx)).is_err() {
        set_last_error("worker thread gone");
        return Err(-2);
    }
    rx.recv().map_err(|_| {
        set_last_error("worker dropped reply");
        -3
    })
}

/// Probe the GPU on the handle's owning thread → adapter description in `*out`.
///
/// # Safety
/// `m` valid; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_model_probe(m: *mut RlModel, out: *mut *mut c_char) -> i32 {
    if out.is_null() {
        set_last_error("null out");
        return -2;
    }
    match call(m, Command::Probe) {
        Ok(Ok(s)) => {
            unsafe { *out = into_c_string(s) };
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(c) => c,
    }
}

/// One-shot probe (create + probe + free).
///
/// # Safety
/// `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_wgpu_probe(out: *mut *mut c_char) -> i32 {
    let m = rl_model_create();
    if m.is_null() {
        return -10;
    }
    let rc = unsafe { rl_model_probe(m, out) };
    unsafe { rl_model_free(m) };
    rc
}

/// Load a GGUF model from a filesystem path. `max_ctx` 0 = default cap.
/// `text_only` skips the vision/audio towers.
///
/// # Safety
/// `m` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_model_load_path(
    m: *mut RlModel,
    path: *const c_char,
    max_ctx: u32,
    text_only: i32,
) -> i32 {
    let path = match unsafe { c_str(path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    match call(m, |reply| Command::LoadPath {
        path,
        max_ctx,
        text_only: text_only != 0,
        reply,
    }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Encode text → token ids. Writes a newly allocated array (free with
/// `rl_free_u32`) to `*out_ids` and its length to `*out_n`.
///
/// # Safety
/// `m` valid; `text` a valid C string; `out_ids`/`out_n` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_encode(
    m: *mut RlModel,
    text: *const c_char,
    out_ids: *mut *mut u32,
    out_n: *mut usize,
) -> i32 {
    if out_ids.is_null() || out_n.is_null() {
        set_last_error("null out");
        return -2;
    }
    let text = match unsafe { c_str(text) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    match call(m, |reply| Command::Encode { text, reply }) {
        Ok(Ok(ids)) => {
            let boxed = ids.into_boxed_slice();
            let n = boxed.len();
            unsafe {
                *out_n = n;
                *out_ids = Box::into_raw(boxed) as *mut u32;
            }
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Decode one token id → its raw vocab string (free with `rl_free_str`).
/// Returns 0 on success, -7 if the id has no entry.
///
/// # Safety
/// `m` valid; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_token_str(m: *mut RlModel, id: u32, out: *mut *mut c_char) -> i32 {
    if out.is_null() {
        set_last_error("null out");
        return -2;
    }
    match call(m, |reply| Command::TokenStr { id, reply }) {
        Ok(Some(s)) => {
            unsafe { *out = into_c_string(s) };
            0
        }
        Ok(None) => -7,
        Err(c) => c,
    }
}

/// Set sampling parameters.
///
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_set_sampling(
    m: *mut RlModel,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
) -> i32 {
    let opts = SamplingOptions {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
    };
    match call(m, |reply| Command::SetSampling { opts, reply }) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// Reset the KV cache / conversation state.
///
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_reset(m: *mut RlModel) -> i32 {
    match call(m, Command::Reset) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// Vocabulary size (0 if no model / error).
///
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_vocab_size(m: *mut RlModel) -> u32 {
    call(m, Command::VocabSize).unwrap_or(0)
}

/// Current KV position (0 if no model / error).
///
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_position(m: *mut RlModel) -> u32 {
    call(m, Command::Position).unwrap_or(0)
}

/// Stream a generation. Feeds `prompt` (prefill) then decodes up to `max_new`
/// tokens, invoking `cb(ctx, token_id, is_eos)` per produced token on the
/// worker thread. Returns the number of tokens produced (>=0), or a negative
/// error code. Use `rl_cancel` to stop early.
///
/// # Safety
/// `m` valid; `prompt`/`n` a valid array; `cb` a valid function pointer; `ctx`
/// kept alive by the caller until this returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_generate(
    m: *mut RlModel,
    prompt: *const u32,
    n: usize,
    max_new: u32,
    cb: TokenFn,
    ctx: *mut c_void,
) -> i32 {
    let prompt = if prompt.is_null() || n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(prompt, n) }.to_vec()
    };
    let cb = TokenCb { f: cb, ctx };
    match call(m, |reply| Command::Generate {
        prompt,
        max_new,
        cb,
        reply,
    }) {
        Ok(Ok(produced)) => produced as i32,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Request cancellation of an in-flight `rl_generate` (sets a shared flag the
/// decode loop polls). Safe to call from any thread.
///
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_cancel(m: *mut RlModel) {
    if let Some(model) = RlModel::from_ptr(m) {
        model.cancel.store(true, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Multimodal (image + audio in)
// ---------------------------------------------------------------------------

/// Frees an `f32` array previously returned by this library.
///
/// # Safety
/// `ptr`/`n` must come from a single returned allocation, unused after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_free_f32(ptr: *mut f32, n: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n)) });
    }
}

fn put_f32(v: Vec<f32>, out: *mut *mut f32, out_len: *mut usize) {
    let boxed = v.into_boxed_slice();
    unsafe {
        *out_len = boxed.len();
        *out = Box::into_raw(boxed) as *mut f32;
    }
}

/// 1 if the loaded model has a vision tower, else 0.
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_has_vision(m: *mut RlModel) -> i32 {
    i32::from(call(m, Command::HasVision).unwrap_or(false))
}

/// 1 if the loaded model has an audio tower, else 0.
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_has_audio(m: *mut RlModel) -> i32 {
    i32::from(call(m, Command::HasAudio).unwrap_or(false))
}

unsafe fn sentinels(m: *mut RlModel, audio: bool, begin: *mut u32, end: *mut u32) -> i32 {
    if begin.is_null() || end.is_null() {
        set_last_error("null out");
        return -2;
    }
    match call(m, |reply| Command::Sentinels { audio, reply }) {
        Ok(Some((b, e))) => {
            unsafe {
                *begin = b;
                *end = e;
            }
            0
        }
        Ok(None) => -7,
        Err(c) => c,
    }
}

/// Image sentinel token ids (begin, end). Returns -7 if absent.
/// # Safety
/// `m` valid; `begin`/`end` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_image_sentinel_ids(
    m: *mut RlModel,
    begin: *mut u32,
    end: *mut u32,
) -> i32 {
    unsafe { sentinels(m, false, begin, end) }
}

/// Audio sentinel token ids (begin, end). Returns -7 if absent.
/// # Safety
/// `m` valid; `begin`/`end` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_audio_sentinel_ids(
    m: *mut RlModel,
    begin: *mut u32,
    end: *mut u32,
) -> i32 {
    unsafe { sentinels(m, true, begin, end) }
}

/// Soft-token count for an h×w image, or -1 if unavailable.
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_image_soft_token_count(m: *mut RlModel, h: usize, w: usize) -> i64 {
    match call(m, |reply| Command::ImageSoftCount { h, w, reply }) {
        Ok(Some(n)) => n as i64,
        _ => -1,
    }
}

/// Encode image pixels (channel-first f32) → soft-token embeddings (free with
/// `rl_free_f32`). Length is `soft_tokens × d_text`.
/// # Safety
/// `m` valid; `pixels`/`n` a valid array; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_encode_image(
    m: *mut RlModel,
    pixels: *const f32,
    n: usize,
    h: usize,
    w: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let pixels = if pixels.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(pixels, n) }.to_vec()
    };
    match call(m, |reply| Command::EncodeImage {
        pixels,
        h,
        w,
        reply,
    }) {
        Ok(Ok(v)) => {
            put_f32(v, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Encode audio PCM (f32 mono) → soft-token embeddings (free with `rl_free_f32`).
/// # Safety
/// `m` valid; `pcm`/`n` a valid array; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_encode_audio(
    m: *mut RlModel,
    pcm: *const f32,
    n: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let pcm = if pcm.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(pcm, n) }.to_vec()
    };
    match call(m, |reply| Command::EncodeAudio { pcm, reply }) {
        Ok(Ok(v)) => {
            put_f32(v, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Decode WAV bytes → mono f32 PCM (free with `rl_free_f32`). Standalone — no
/// model needed.
/// # Safety
/// `bytes`/`n` a valid array; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_decode_wav(
    bytes: *const u8,
    n: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if bytes.is_null() || out.is_null() || out_len.is_null() {
        set_last_error("null");
        return -2;
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes, n) };
    match Model::decode_wav_native(slice) {
        Ok(v) => {
            put_f32(v, out, out_len);
            0
        }
        Err(e) => {
            set_last_error(format!("{e}"));
            -6
        }
    }
}

/// Streaming generation with one spliced media item: after the `sentinel_begin`
/// token is fed during prefill, the `soft` rows (`soft_len/d_text` of them) are
/// spliced via step_with_embedding. Returns produced count (>=0) or negative.
/// # Safety
/// `m` valid; `prompt`/`n` and `soft`/`soft_len` valid arrays; `cb` valid; `ctx`
/// kept alive until return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_generate_spliced(
    m: *mut RlModel,
    prompt: *const u32,
    n: usize,
    sentinel_begin: u32,
    soft: *const f32,
    soft_len: usize,
    d_text: usize,
    max_new: u32,
    cb: TokenFn,
    ctx: *mut c_void,
) -> i32 {
    let prompt = if prompt.is_null() || n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(prompt, n) }.to_vec()
    };
    let soft = if soft.is_null() || soft_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(soft, soft_len) }.to_vec()
    };
    let cb = TokenCb { f: cb, ctx };
    match call(m, |reply| Command::GenerateSpliced {
        prompt,
        sentinel_begin,
        soft,
        d_text,
        max_new,
        cb,
        reply,
    }) {
        Ok(Ok(produced)) => produced as i32,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

// ---------------------------------------------------------------------------
// Text-to-speech (Kokoro) — a SEPARATE model handle with its own owning thread.
// ---------------------------------------------------------------------------

const TTS_SAMPLE_RATE: u32 = 24_000;

enum TtsCommand {
    Load {
        bytes: Vec<u8>,
        reply: Sender<Result<(), String>>,
    },
    SetLexicon {
        gold: Vec<u8>,
        silver: Vec<u8>,
        reply: Sender<()>,
    },
    Synthesize {
        text: String,
        voice: String,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    Shutdown,
}

fn tts_worker(rx: mpsc::Receiver<TtsCommand>) {
    let mut tts: Option<KokoroTts> = None;
    for cmd in rx {
        match cmd {
            TtsCommand::Load { bytes, reply } => {
                let r =
                    pollster::block_on(KokoroTts::load_native(bytes)).map_err(|e| format!("{e}"));
                match r {
                    Ok(t) => {
                        tts = Some(t);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            TtsCommand::SetLexicon {
                gold,
                silver,
                reply,
            } => {
                if let Some(t) = tts.as_mut() {
                    t.set_lexicon_native(&gold, &silver);
                }
                let _ = reply.send(());
            }
            TtsCommand::Synthesize { text, voice, reply } => {
                let r = match tts.as_mut() {
                    Some(t) => {
                        let (pcm, _oov) =
                            pollster::block_on(t.synthesize_native(&text, &voice, None));
                        Ok(pcm)
                    }
                    None => Err("tts not loaded".into()),
                };
                let _ = reply.send(r);
            }
            TtsCommand::Shutdown => break,
        }
    }
}

/// Opaque TTS handle (Kokoro). Owns its worker thread.
pub struct RlTts {
    tx: Sender<TtsCommand>,
    handle: Option<JoinHandle<()>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_tts_create() -> *mut RlTts {
    let (tx, rx) = mpsc::channel();
    match std::thread::Builder::new()
        .name("rullama-tts".into())
        .spawn(move || tts_worker(rx))
    {
        Ok(handle) => Box::into_raw(Box::new(RlTts {
            tx,
            handle: Some(handle),
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn tts thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `t` must be NULL or a handle from `rl_tts_create`, unused afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_free(t: *mut RlTts) {
    if t.is_null() {
        return;
    }
    let mut b = unsafe { Box::from_raw(t) };
    let _ = b.tx.send(TtsCommand::Shutdown);
    if let Some(h) = b.handle.take() {
        let _ = h.join();
    }
}

fn tts_call<T>(t: *mut RlTts, make: impl FnOnce(Sender<T>) -> TtsCommand) -> Result<T, i32> {
    let Some(tts) = (unsafe { t.as_ref() }) else {
        set_last_error("null tts handle");
        return Err(-1);
    };
    let (tx, rx) = mpsc::channel();
    if tts.tx.send(make(tx)).is_err() {
        set_last_error("tts worker gone");
        return Err(-2);
    }
    rx.recv().map_err(|_| {
        set_last_error("tts dropped reply");
        -3
    })
}

/// Load the Kokoro GGUF from a path.
/// # Safety
/// `t` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_load_path(t: *mut RlTts, path: *const c_char) -> i32 {
    let path = match unsafe { c_str(path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path}: {e}"));
            return -6;
        }
    };
    match tts_call(t, |reply| TtsCommand::Load { bytes, reply }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Set the G2P lexicon from gold (+ optional silver) JSON file paths.
/// # Safety
/// `t` valid; paths valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_set_lexicon(
    t: *mut RlTts,
    gold_path: *const c_char,
    silver_path: *const c_char,
) -> i32 {
    let gp = match unsafe { c_str(gold_path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let sp = match unsafe { c_str(silver_path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let gold = std::fs::read(&gp).unwrap_or_default();
    let silver = std::fs::read(&sp).unwrap_or_default();
    match tts_call(t, |reply| TtsCommand::SetLexicon {
        gold,
        silver,
        reply,
    }) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// Synthesize text → mono f32 PCM at `rl_tts_sample_rate` (free with `rl_free_f32`).
/// # Safety
/// `t` valid; `text`/`voice` valid C strings; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_synthesize(
    t: *mut RlTts,
    text: *const c_char,
    voice: *const c_char,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let text = match unsafe { c_str(text) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let voice = match unsafe { c_str(voice) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    match tts_call(t, |reply| TtsCommand::Synthesize { text, voice, reply }) {
        Ok(Ok(pcm)) => {
            put_f32(pcm, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// TTS PCM sample rate (Hz).
/// # Safety
/// `t` may be NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rl_tts_sample_rate(_t: *mut RlTts) -> u32 {
    TTS_SAMPLE_RATE
}

// ---------------------------------------------------------------------------
// Voice cloning (StyleTTS2) — a SEPARATE model handle with its own thread.
// ---------------------------------------------------------------------------

enum CloneCommand {
    Load {
        bytes: Vec<u8>,
        reply: Sender<Result<(), String>>,
    },
    SetLexicon {
        gold: Vec<u8>,
        silver: Vec<u8>,
        reply: Sender<()>,
    },
    EncodeVoice {
        pcm: Vec<f32>,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    Synthesize {
        text: String,
        voice: Vec<f32>,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    Shutdown,
}

fn clone_worker(rx: mpsc::Receiver<CloneCommand>) {
    let mut clone: Option<StyleTtsClone> = None;
    for cmd in rx {
        match cmd {
            CloneCommand::Load { bytes, reply } => {
                let r = pollster::block_on(StyleTtsClone::load_native(bytes))
                    .map_err(|e| format!("{e}"));
                match r {
                    Ok(c) => {
                        clone = Some(c);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            CloneCommand::SetLexicon {
                gold,
                silver,
                reply,
            } => {
                if let Some(c) = clone.as_mut() {
                    c.set_lexicon_native(&gold, &silver);
                }
                let _ = reply.send(());
            }
            CloneCommand::EncodeVoice { pcm, reply } => {
                let r = match clone.as_mut() {
                    Some(c) => Ok(pollster::block_on(c.encode_voice_native(&pcm, None))),
                    None => Err("clone model not loaded".into()),
                };
                let _ = reply.send(r);
            }
            CloneCommand::Synthesize { text, voice, reply } => {
                let r = match clone.as_mut() {
                    Some(c) => Ok(pollster::block_on(c.synthesize_native(&text, &voice, None))),
                    None => Err("clone model not loaded".into()),
                };
                let _ = reply.send(r);
            }
            CloneCommand::Shutdown => break,
        }
    }
}

/// Opaque voice-clone handle (StyleTTS2).
pub struct RlClone {
    tx: Sender<CloneCommand>,
    handle: Option<JoinHandle<()>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_clone_create() -> *mut RlClone {
    let (tx, rx) = mpsc::channel();
    match std::thread::Builder::new()
        .name("rullama-clone".into())
        .spawn(move || clone_worker(rx))
    {
        Ok(handle) => Box::into_raw(Box::new(RlClone {
            tx,
            handle: Some(handle),
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn clone thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `t` must be NULL or a handle from `rl_clone_create`, unused afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clone_free(t: *mut RlClone) {
    if t.is_null() {
        return;
    }
    let mut b = unsafe { Box::from_raw(t) };
    let _ = b.tx.send(CloneCommand::Shutdown);
    if let Some(h) = b.handle.take() {
        let _ = h.join();
    }
}

fn clone_call<T>(t: *mut RlClone, make: impl FnOnce(Sender<T>) -> CloneCommand) -> Result<T, i32> {
    let Some(c) = (unsafe { t.as_ref() }) else {
        set_last_error("null clone handle");
        return Err(-1);
    };
    let (tx, rx) = mpsc::channel();
    if c.tx.send(make(tx)).is_err() {
        set_last_error("clone worker gone");
        return Err(-2);
    }
    rx.recv().map_err(|_| {
        set_last_error("clone dropped reply");
        -3
    })
}

/// Load the StyleTTS2 GGUF from a path.
/// # Safety
/// `t` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clone_load_path(t: *mut RlClone, path: *const c_char) -> i32 {
    let path = match unsafe { c_str(path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path}: {e}"));
            return -6;
        }
    };
    match clone_call(t, |reply| CloneCommand::Load { bytes, reply }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Set the G2P lexicon from gold + silver JSON file paths.
/// # Safety
/// `t` valid; paths valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clone_set_lexicon(
    t: *mut RlClone,
    gold_path: *const c_char,
    silver_path: *const c_char,
) -> i32 {
    let gp = match unsafe { c_str(gold_path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let sp = match unsafe { c_str(silver_path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let gold = std::fs::read(&gp).unwrap_or_default();
    let silver = std::fs::read(&sp).unwrap_or_default();
    match clone_call(t, |reply| CloneCommand::SetLexicon {
        gold,
        silver,
        reply,
    }) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// Encode a 24 kHz mono reference clip → a speaker-voice vector (free with `rl_free_f32`).
/// # Safety
/// `t` valid; `pcm`/`n` a valid array; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clone_encode_voice(
    t: *mut RlClone,
    pcm: *const f32,
    n: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let pcm = if pcm.is_null() || n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(pcm, n) }.to_vec()
    };
    match clone_call(t, |reply| CloneCommand::EncodeVoice { pcm, reply }) {
        Ok(Ok(v)) => {
            put_f32(v, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Synthesize text with a cloned voice vector → 24 kHz PCM (free with `rl_free_f32`).
/// # Safety
/// `t` valid; `text` a valid C string; `voice`/`voice_len` a valid array; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clone_synthesize(
    t: *mut RlClone,
    text: *const c_char,
    voice: *const f32,
    voice_len: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let text = match unsafe { c_str(text) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let voice = if voice.is_null() || voice_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(voice, voice_len) }.to_vec()
    };
    match clone_call(t, |reply| CloneCommand::Synthesize { text, voice, reply }) {
        Ok(Ok(pcm)) => {
            put_f32(pcm, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Clone TTS PCM sample rate (Hz).
/// # Safety
/// `t` may be NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rl_clone_sample_rate(_t: *mut RlClone) -> u32 {
    TTS_SAMPLE_RATE
}

// ---------------------------------------------------------------------------
// Embeddings (EmbeddingGemma) — a SEPARATE model handle with its own thread.
// ---------------------------------------------------------------------------

enum EmbedCommand {
    Load {
        bytes: Vec<u8>,
        reply: Sender<Result<(), String>>,
    },
    Dim(Sender<u32>),
    Embed {
        text: String,
        target_dim: usize,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    Shutdown,
}

fn embed_worker(rx: mpsc::Receiver<EmbedCommand>) {
    let mut model: Option<EmbeddingModel> = None;
    for cmd in rx {
        match cmd {
            EmbedCommand::Load { bytes, reply } => {
                let r = pollster::block_on(EmbeddingModel::load_native(bytes))
                    .map_err(|e| format!("{e}"));
                match r {
                    Ok(m) => {
                        model = Some(m);
                        let _ = reply.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                }
            }
            EmbedCommand::Dim(reply) => {
                let _ = reply.send(model.as_ref().map_or(0, |m| m.dim_native()));
            }
            EmbedCommand::Embed {
                text,
                target_dim,
                reply,
            } => {
                let r = match model.as_ref() {
                    Some(m) => pollster::block_on(m.embed_native(&text, target_dim))
                        .map_err(|e| format!("{e}")),
                    None => Err("embedding model not loaded".into()),
                };
                let _ = reply.send(r);
            }
            EmbedCommand::Shutdown => break,
        }
    }
}

/// Opaque embedding-model handle.
pub struct RlEmbed {
    tx: Sender<EmbedCommand>,
    handle: Option<JoinHandle<()>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_embed_create() -> *mut RlEmbed {
    let (tx, rx) = mpsc::channel();
    match std::thread::Builder::new()
        .name("rullama-embed".into())
        .spawn(move || embed_worker(rx))
    {
        Ok(handle) => Box::into_raw(Box::new(RlEmbed {
            tx,
            handle: Some(handle),
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn embed thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `t` must be NULL or a handle from `rl_embed_create`, unused afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed_free(t: *mut RlEmbed) {
    if t.is_null() {
        return;
    }
    let mut b = unsafe { Box::from_raw(t) };
    let _ = b.tx.send(EmbedCommand::Shutdown);
    if let Some(h) = b.handle.take() {
        let _ = h.join();
    }
}

fn embed_call<T>(t: *mut RlEmbed, make: impl FnOnce(Sender<T>) -> EmbedCommand) -> Result<T, i32> {
    let Some(e) = (unsafe { t.as_ref() }) else {
        set_last_error("null embed handle");
        return Err(-1);
    };
    let (tx, rx) = mpsc::channel();
    if e.tx.send(make(tx)).is_err() {
        set_last_error("embed worker gone");
        return Err(-2);
    }
    rx.recv().map_err(|_| {
        set_last_error("embed dropped reply");
        -3
    })
}

/// Load the embedding GGUF from a path.
/// # Safety
/// `t` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed_load_path(t: *mut RlEmbed, path: *const c_char) -> i32 {
    let path = match unsafe { c_str(path) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path}: {e}"));
            return -6;
        }
    };
    match embed_call(t, |reply| EmbedCommand::Load { bytes, reply }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Native embedding dimension (0 if not loaded).
/// # Safety
/// `t` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed_dim(t: *mut RlEmbed) -> u32 {
    embed_call(t, EmbedCommand::Dim).unwrap_or(0)
}

/// Embed text → f32 vector of length `target_dim` (Matryoshka; free with `rl_free_f32`).
/// # Safety
/// `t` valid; `text` a valid C string; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed(
    t: *mut RlEmbed,
    text: *const c_char,
    target_dim: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    let text = match unsafe { c_str(text) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    match embed_call(t, |reply| EmbedCommand::Embed {
        text,
        target_dim,
        reply,
    }) {
        Ok(Ok(v)) => {
            put_f32(v, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

// ---------------------------------------------------------------------------
// Fine-tuning (LoRA) — reuses the model handle (the Model becomes a trainer).
// ---------------------------------------------------------------------------

/// Frees a byte array previously returned by this library (e.g. adapter bytes).
/// # Safety
/// `ptr`/`n` must come from a single returned allocation, unused after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_free_bytes(ptr: *mut u8, n: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n)) });
    }
}

fn put_bytes(v: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) {
    let boxed = v.into_boxed_slice();
    unsafe {
        *out_len = boxed.len();
        *out = Box::into_raw(boxed) as *mut u8;
    }
}

/// Convert the loaded model into a LoRA training session (consumes the model).
/// `target_modules` is a comma-separated list (empty → defaults).
/// # Safety
/// `m` valid; `target_modules` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_trainer_begin(
    m: *mut RlModel,
    rank: u32,
    alpha: f32,
    dropout: f32,
    target_modules: *const c_char,
    max_seq_len: usize,
    learning_rate: f64,
) -> i32 {
    let mods = match unsafe { c_str(target_modules) } {
        Ok(s) => s
            .split(',')
            .map(|x| x.trim().to_owned())
            .filter(|x| !x.is_empty())
            .collect::<Vec<_>>(),
        Err(_) => Vec::new(),
    };
    match call(m, |reply| Command::TrainerBegin {
        rank,
        alpha,
        dropout,
        target_modules: mods,
        max_seq_len,
        learning_rate,
        reply,
    }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// One training step on (input_ids → target). Writes the loss to `*out_loss`.
/// # Safety
/// `m` valid; `input_ids`/`n` a valid array; `out_loss` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_trainer_step(
    m: *mut RlModel,
    input_ids: *const u32,
    n: usize,
    target: u32,
    out_loss: *mut f32,
) -> i32 {
    if out_loss.is_null() {
        set_last_error("null out");
        return -2;
    }
    let ids = if input_ids.is_null() || n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(input_ids, n) }.to_vec()
    };
    match call(m, |reply| Command::TrainerStep {
        input_ids: ids,
        target,
        reply,
    }) {
        Ok(Ok(loss)) => {
            unsafe { *out_loss = loss };
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Serialize the trained LoRA adapter → safetensors bytes (free with `rl_free_bytes`).
/// # Safety
/// `m` valid; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_trainer_save_adapter(
    m: *mut RlModel,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out.is_null() || out_len.is_null() {
        set_last_error("null out");
        return -2;
    }
    match call(m, |reply| Command::TrainerSaveAdapter { reply }) {
        Ok(Ok(bytes)) => {
            put_bytes(bytes, out, out_len);
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Load a LoRA adapter (safetensors bytes) into the chat model. Returns slot
/// count (>=0) or negative on error.
/// # Safety
/// `m` valid; `bytes`/`n` a valid array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_load_adapter(m: *mut RlModel, bytes: *const u8, n: usize) -> i32 {
    let bytes = if bytes.is_null() || n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(bytes, n) }.to_vec()
    };
    match call(m, |reply| Command::LoadAdapter { bytes, reply }) {
        Ok(Ok(slots)) => slots as i32,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

/// Clear any active LoRA adapter from the chat model.
/// # Safety
/// `m` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_clear_adapter(m: *mut RlModel) -> i32 {
    match call(m, Command::ClearAdapter) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// ROME knowledge edit (mutates the chat model): make `subject` within `prompt`
/// predict `target` at `layer`. Slow (iterative gradient). Returns 0 on success.
/// # Safety
/// `m` valid; `prompt`/`subject`/`target` valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_rome_edit(
    m: *mut RlModel,
    prompt: *const c_char,
    subject: *const c_char,
    target: *const c_char,
    layer: u32,
) -> i32 {
    let prompt = match unsafe { c_str(prompt) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let subject = match unsafe { c_str(subject) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let target = match unsafe { c_str(target) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    match call(m, |reply| Command::RomeEdit {
        prompt,
        subject,
        target,
        layer,
        reply,
    }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(c) => c,
    }
}

// ---------------------------------------------------------------------------
// M12: Rhai tool-orchestration (memoize-and-replay across the FFI boundary)
// ---------------------------------------------------------------------------
//
// The model writes a Rhai script that calls tools (e.g. `get_weather("Tokyo")`).
// Tool *execution* lives on the C# side (async HTTP, registry), so we cannot run
// it inside Rhai. Instead, each registered tool is bound to a Rhai function that
// looks up a result **cache**; on a miss it records the call and aborts. The C#
// caller resolves the missing call(s), adds them to the cache, and calls again —
// converging in ≤depth passes (data-dependent branches re-evaluate on real
// values). Returns a JSON envelope:
//
//   {"status":"needed","needed":[{"name":"get_weather","arg":"Tokyo"}]}
//   {"status":"final","final":"<result string>"}
//   {"status":"error","error":"<message>"}
//
// Cache key: `"<name>\u{1}<arg>"`. Cache values are arbitrary JSON (objects map
// to Rhai maps so the model can read `.temp_c`, strings stay strings, etc.).

fn json_to_dynamic(v: &serde_json::Value) -> rhai::Dynamic {
    use serde_json::Value;
    match v {
        Value::Null => rhai::Dynamic::UNIT,
        Value::Bool(b) => (*b).into(),
        Value::Number(n) => n
            .as_i64()
            .map(rhai::Dynamic::from)
            .unwrap_or_else(|| n.as_f64().unwrap_or(0.0).into()),
        Value::String(s) => s.clone().into(),
        Value::Array(a) => a
            .iter()
            .map(json_to_dynamic)
            .collect::<rhai::Array>()
            .into(),
        Value::Object(o) => {
            let mut m = rhai::Map::new();
            for (k, val) in o {
                m.insert(k.as_str().into(), json_to_dynamic(val));
            }
            m.into()
        }
    }
}

const ORCH_MISS_TAG: &str = "__rl_tool_miss__";

fn run_orchestrator(script: &str, cached_json: &str, tool_names: &str) -> serde_json::Value {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Parse the cache: { "<name>\u{1}<arg>": <json value> }.
    let cache: HashMap<String, serde_json::Value> = if cached_json.trim().is_empty() {
        HashMap::new()
    } else {
        match serde_json::from_str(cached_json) {
            Ok(m) => m,
            Err(e) => {
                return serde_json::json!({"status":"error","error":format!("bad cache json: {e}")});
            }
        }
    };
    let cache = Rc::new(cache);
    let missed: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));

    let mut engine = rhai::Engine::new();
    engine.set_max_operations(2_000_000);
    engine.set_max_call_levels(64);

    for name in tool_names
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let name = name.to_string();
        // arity-1: tool("arg")
        {
            let name = name.clone();
            let cache = Rc::clone(&cache);
            let missed = Rc::clone(&missed);
            engine.register_fn(
                name.clone(),
                move |arg: rhai::ImmutableString| -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
                    let key = format!("{name}\u{1}{arg}");
                    if let Some(v) = cache.get(&key) {
                        Ok(json_to_dynamic(v))
                    } else {
                        missed.borrow_mut().push((name.clone(), arg.to_string()));
                        Err(ORCH_MISS_TAG.into())
                    }
                },
            );
        }
        // arity-0: tool()
        {
            let name = name.clone();
            let cache = Rc::clone(&cache);
            let missed = Rc::clone(&missed);
            engine.register_fn(
                name.clone(),
                move || -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
                    let key = format!("{name}\u{1}");
                    if let Some(v) = cache.get(&key) {
                        Ok(json_to_dynamic(v))
                    } else {
                        missed.borrow_mut().push((name.clone(), String::new()));
                        Err(ORCH_MISS_TAG.into())
                    }
                },
            );
        }
    }

    let result = engine.eval::<rhai::Dynamic>(script);

    let pending = missed.borrow();
    if !pending.is_empty() {
        let needed: Vec<serde_json::Value> = pending
            .iter()
            .map(|(n, a)| serde_json::json!({"name": n, "arg": a}))
            .collect();
        return serde_json::json!({"status":"needed","needed": needed});
    }
    match result {
        Ok(v) => serde_json::json!({"status":"final","final": v.to_string()}),
        Err(e) => serde_json::json!({"status":"error","error": e.to_string()}),
    }
}

/// Run a model-authored Rhai orchestration script. See the module comment above
/// for the cache/replay protocol. Writes a JSON envelope to `out` (free with
/// `rl_free_str`). Returns 0 on success, negative on a null/utf-8 argument.
///
/// # Safety
/// `script`/`cached_json`/`tool_names` valid C strings; `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_orch_run(
    script: *const c_char,
    cached_json: *const c_char,
    tool_names: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    if out.is_null() {
        set_last_error("null out");
        return -2;
    }
    let script = match unsafe { c_str(script) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let cached = match unsafe { c_str(cached_json) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let names = match unsafe { c_str(tool_names) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let envelope = run_orchestrator(script, cached, names);
    unsafe { *out = into_c_string(envelope.to_string()) };
    0
}

// ---------------------------------------------------------------------------
// M13: image generation (DiT diffusion) — native, over imagegen::ImageBundle
// ---------------------------------------------------------------------------
//
// The published `imagegen` engine (Qwen3 encoder → S3-DiT denoise loop w/ CFG →
// VAE decode) runs natively via `ImageBundle<FileBlobSource>` — the exact
// pipeline the browser's `ImageModel` wraps. The handle owns one OS thread (wgpu
// state is `!Send`); commands are marshalled over an MPSC channel. The model dir
// is a standard Z-Image layout: `text_encoder/ transformer/ vae/ tokenizer/`.

/// Per-step progress callback: `(ctx, step, total, stage)`. `stage` ("encode"/
/// "denoise"/"vae") is valid only for the duration of the call.
type ImageProgressFn = extern "C" fn(ctx: *mut c_void, step: u32, total: u32, stage: *const c_char);

/// `Send` wrapper so the progress callback + ctx can cross the command channel.
struct ImgProgressCb {
    f: ImageProgressFn,
    ctx: *mut c_void,
}
unsafe impl Send for ImgProgressCb {}

/// Generated image reply: raw RGBA bytes plus width and height, or an error.
type ImgGenResult = Result<(Vec<u8>, u32, u32), String>;

enum ImgCommand {
    Load {
        dir: String,
        reply: Sender<Result<(), String>>,
    },
    Generate {
        prompt: String,
        neg: String,
        cfg_scale: f32,
        lh: usize,
        lw: usize,
        steps: usize,
        seed: u64,
        cb: ImgProgressCb,
        reply: Sender<ImgGenResult>,
    },
    Shutdown,
}

fn imagegen_worker(rx: mpsc::Receiver<ImgCommand>) {
    // `!Send` engine state — created and dropped only on this thread.
    let mut bundle: Option<ImageBundle<FileBlobSource>> = None;
    let mut tokenizer: Option<Tokenizer> = None;
    let mut down: usize = 8;

    for cmd in rx {
        match cmd {
            ImgCommand::Load { dir, reply } => {
                let res = (|| -> Result<(), String> {
                    let vae_bytes = std::fs::read(format!("{dir}/vae/config.json"))
                        .map_err(|e| format!("read vae config: {e}"))?;
                    down = VaeConfig::parse(&vae_bytes)
                        .map_err(|e| format!("parse vae config: {e}"))?
                        .downscale() as usize;
                    let tk = Tokenizer::from_file(format!("{dir}/tokenizer/tokenizer.json"))
                        .map_err(|e| format!("load tokenizer: {e}"))?;
                    let b = pollster::block_on(ImageBundle::open(
                        FileBlobSource::new(format!("{dir}/text_encoder")),
                        FileBlobSource::new(format!("{dir}/transformer")),
                        FileBlobSource::new(format!("{dir}/vae")),
                    ))
                    .map_err(|e| format!("open bundle: {e}"))?;
                    bundle = Some(b);
                    tokenizer = Some(tk);
                    Ok(())
                })();
                let _ = reply.send(res);
            }
            ImgCommand::Generate {
                prompt,
                neg,
                cfg_scale,
                lh,
                lw,
                steps,
                seed,
                cb,
                reply,
            } => {
                let res = (|| -> Result<(Vec<u8>, u32, u32), String> {
                    let b = bundle.as_ref().ok_or("no image model loaded")?;
                    let tk = tokenizer.as_ref().ok_or("no tokenizer loaded")?;
                    // Wrap in the chat format the Qwen encoder expects (mirrors the
                    // browser path + the imagegen_generate_gpu example).
                    let wrap = |p: &str| {
                        format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
                    };
                    let tokens: Vec<u32> = tk
                        .encode(wrap(&prompt), false)
                        .map_err(|e| format!("encode prompt: {e}"))?
                        .get_ids()
                        .to_vec();
                    let neg_tokens: Vec<u32> = if neg.is_empty() {
                        Vec::new()
                    } else {
                        tk.encode(wrap(&neg), false)
                            .map_err(|e| format!("encode neg: {e}"))?
                            .get_ids()
                            .to_vec()
                    };
                    let prog = |stage: &str, i: usize, n: usize| {
                        if let Ok(cs) = CString::new(stage) {
                            (cb.f)(cb.ctx, i as u32, n as u32, cs.as_ptr());
                        }
                    };
                    let rgb = pollster::block_on(b.generate(
                        &tokens,
                        &neg_tokens,
                        cfg_scale,
                        lh,
                        lw,
                        steps,
                        seed,
                        Some(&prog),
                    ))
                    .map_err(|e| format!("generate: {e}"))?;
                    let (h, w) = (lh * down, lw * down);
                    let rgba = rgb_chw_to_rgba8(&rgb, h, w);
                    Ok((rgba, w as u32, h as u32))
                })();
                let _ = reply.send(res);
            }
            ImgCommand::Shutdown => break,
        }
    }
}

/// Opaque image-generation handle: owns the worker thread + command channel.
pub struct RlImageGen {
    tx: Sender<ImgCommand>,
    handle: Option<JoinHandle<()>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_imagegen_create() -> *mut RlImageGen {
    let (tx, rx) = mpsc::channel();
    match std::thread::Builder::new()
        .name("rullama-imagegen".into())
        .spawn(move || imagegen_worker(rx))
    {
        Ok(handle) => Box::into_raw(Box::new(RlImageGen {
            tx,
            handle: Some(handle),
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn imagegen thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `t` must be NULL or a handle from `rl_imagegen_create`, unused after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_imagegen_free(t: *mut RlImageGen) {
    if t.is_null() {
        return;
    }
    let mut g = unsafe { Box::from_raw(t) };
    let _ = g.tx.send(ImgCommand::Shutdown);
    if let Some(h) = g.handle.take() {
        let _ = h.join();
    }
}

/// Load a Z-Image model directory (`text_encoder/ transformer/ vae/ tokenizer/`).
/// Returns 0 on success, negative on error (`rl_last_error` has the detail).
///
/// # Safety
/// `t` valid; `dir` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_imagegen_load_blobs(t: *mut RlImageGen, dir: *const c_char) -> i32 {
    let Some(g) = (unsafe { t.as_ref() }) else {
        set_last_error("null handle");
        return -2;
    };
    let dir = match unsafe { c_str(dir) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let (tx, rx) = mpsc::channel();
    if g.tx.send(ImgCommand::Load { dir, reply: tx }).is_err() {
        set_last_error("imagegen worker gone");
        return -3;
    }
    match rx.recv() {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(_) => {
            set_last_error("imagegen worker dropped reply");
            -3
        }
    }
}

/// Generate an image. Writes RGBA8 bytes (`*out`/`*out_len`, free with
/// `rl_free_bytes`) plus pixel dimensions (`*out_w`/`*out_h`), and calls
/// `progress` per encode/denoise/VAE step. `latent_h`/`latent_w` are in latent
/// cells (image = latent × VAE downscale, typically 8×).
///
/// # Safety
/// `t` valid; `prompt`/`neg_prompt` valid C strings; out pointers writable;
/// `progress` callable with `ctx` for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_imagegen_generate(
    t: *mut RlImageGen,
    prompt: *const c_char,
    neg_prompt: *const c_char,
    cfg_scale: f32,
    latent_h: u32,
    latent_w: u32,
    steps: u32,
    seed: u64,
    progress: ImageProgressFn,
    ctx: *mut c_void,
    out: *mut *mut u8,
    out_len: *mut usize,
    out_w: *mut u32,
    out_h: *mut u32,
) -> i32 {
    let Some(g) = (unsafe { t.as_ref() }) else {
        set_last_error("null handle");
        return -2;
    };
    if out.is_null() || out_len.is_null() || out_w.is_null() || out_h.is_null() {
        set_last_error("null out");
        return -2;
    }
    let prompt = match unsafe { c_str(prompt) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            set_last_error(e);
            return -5;
        }
    };
    let neg = unsafe { c_str(neg_prompt) }
        .map(str::to_owned)
        .unwrap_or_default();
    let (tx, rx) = mpsc::channel();
    let cmd = ImgCommand::Generate {
        prompt,
        neg,
        cfg_scale,
        lh: latent_h as usize,
        lw: latent_w as usize,
        steps: steps as usize,
        seed,
        cb: ImgProgressCb { f: progress, ctx },
        reply: tx,
    };
    if g.tx.send(cmd).is_err() {
        set_last_error("imagegen worker gone");
        return -3;
    }
    match rx.recv() {
        Ok(Ok((bytes, w, h))) => {
            put_bytes(bytes, out, out_len);
            unsafe {
                *out_w = w;
                *out_h = h;
            }
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -6
        }
        Err(_) => {
            set_last_error("imagegen worker dropped reply");
            -3
        }
    }
}

// ---------------------------------------------------------------------------
// Native smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_an_adapter() {
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { rl_wgpu_probe(&mut out) };
        assert_eq!(rc, 0, "probe failed: rc={rc}");
        let desc = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        eprintln!("probe: {desc}");
        assert!(desc.contains("adapter="));
        unsafe { rl_free_str(out) };
    }

    #[test]
    fn create_free_roundtrip() {
        let m = rl_model_create();
        assert!(!m.is_null());
        unsafe { rl_model_free(m) };
    }

    // ---- M12: Rhai orchestration (no model needed) ----

    #[test]
    fn orch_reports_first_miss() {
        let env = run_orchestrator("let w = get_weather(\"Tokyo\"); w", "", "get_weather");
        assert_eq!(env["status"], "needed");
        assert_eq!(env["needed"][0]["name"], "get_weather");
        assert_eq!(env["needed"][0]["arg"], "Tokyo");
    }

    #[test]
    fn orch_final_from_cache() {
        // Cache key is "<name>\u{1}<arg>" — the separator must be a JSON-escaped
        // control char (\u0001), exactly as the C# side serializes it.
        let cache = r#"{"get_weather\u0001Tokyo":{"summary":"warm and clear"}}"#;
        let env = run_orchestrator(
            "let w = get_weather(\"Tokyo\"); w.summary",
            cache,
            "get_weather",
        );
        assert_eq!(env["status"], "final");
        assert_eq!(env["final"], "warm and clear");
    }

    #[test]
    fn orch_dependent_branch_reevaluates() {
        // First pass: needs the weather. Second pass (temp cached): the branch is
        // taken on the REAL value and discovers the air-quality call.
        let names = "get_weather,get_air_quality";
        let script = "let w = get_weather(\"Tokyo\");\nif w.temp_c > 20 { get_air_quality(\"Tokyo\") } else { \"cool\" }";

        let e1 = run_orchestrator(script, "", names);
        assert_eq!(e1["status"], "needed");
        assert_eq!(e1["needed"][0]["name"], "get_weather");

        let cache = r#"{"get_weather\u0001Tokyo":{"temp_c":25}}"#;
        let e2 = run_orchestrator(script, cache, names);
        assert_eq!(
            e2["status"], "needed",
            "branch should re-evaluate to air quality"
        );
        assert_eq!(e2["needed"][0]["name"], "get_air_quality");

        let cache2 =
            r#"{"get_weather\u0001Tokyo":{"temp_c":25},"get_air_quality\u0001Tokyo":"good"}"#;
        let e3 = run_orchestrator(script, cache2, names);
        assert_eq!(e3["status"], "final");
        assert_eq!(e3["final"], "good");
    }

    #[test]
    fn orch_syntax_error_surfaces() {
        // Lua-style `then/end` is invalid Rhai → error status (C# falls back).
        let env = run_orchestrator("if true then 1 else 2 end", "", "get_weather");
        assert_eq!(env["status"], "error");
    }

    // ---- M13: image-gen handle plumbing (no weights needed) ----

    #[test]
    fn imagegen_handle_lifecycle_and_error_path() {
        let g = rl_imagegen_create();
        assert!(!g.is_null());
        // Loading a missing model dir must fail gracefully (no panic) with a
        // message marshalled back across the worker channel.
        let bad = CString::new("/nonexistent/z-image-turbo").unwrap();
        let rc = unsafe { rl_imagegen_load_blobs(g, bad.as_ptr()) };
        assert_ne!(rc, 0, "loading a missing dir should fail");
        assert!(!last_error_str().is_empty(), "error message should be set");
        eprintln!("expected load failure: {}", last_error_str());
        unsafe { rl_imagegen_free(g) }; // exercises Shutdown + thread join
    }

    // End-to-end load + generate. Skipped unless RULLAMA_TEST_GGUF points at a
    // Gemma 4 GGUF (e.g. an Ollama e2b blob).
    #[test]
    fn load_and_generate_if_model_available() {
        let Ok(path) = std::env::var("RULLAMA_TEST_GGUF") else {
            eprintln!("skip: set RULLAMA_TEST_GGUF to run");
            return;
        };
        let m = rl_model_create();
        assert!(!m.is_null());
        let cpath = CString::new(path).unwrap();
        // text-only, small context for a fast smoke.
        let rc = unsafe { rl_model_load_path(m, cpath.as_ptr(), 512, 1) };
        assert_eq!(rc, 0, "load failed: {}", last_error_str());

        unsafe { rl_set_sampling(m, 0.0, 0, 1.0, 1.0, 0) }; // greedy

        let prompt =
            "<start_of_turn>user\nSay hello in one word.<end_of_turn>\n<start_of_turn>model\n";
        let cprompt = CString::new(prompt).unwrap();
        let mut ids: *mut u32 = std::ptr::null_mut();
        let mut nn: usize = 0;
        let rc = unsafe { rl_encode(m, cprompt.as_ptr(), &mut ids, &mut nn) };
        assert_eq!(rc, 0, "encode failed: {}", last_error_str());
        assert!(nn > 0);

        extern "C" fn on_tok(ctx: *mut c_void, tok: u32, piece: *const c_char, _eos: i32) {
            let count = unsafe { &mut *(ctx as *mut u32) };
            *count += 1;
            let s = if piece.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(piece) }
                    .to_string_lossy()
                    .into_owned()
            };
            eprintln!("tok #{count}: {tok} = {s:?}");
        }
        let mut count: u32 = 0;
        let produced = unsafe {
            rl_generate(
                m,
                ids,
                nn,
                16,
                on_tok,
                &mut count as *mut u32 as *mut c_void,
            )
        };
        assert!(produced > 0, "no tokens produced: {}", last_error_str());
        assert_eq!(produced as u32, count);

        unsafe { rl_free_u32(ids, nn) };
        unsafe { rl_model_free(m) };
    }

    fn last_error_str() -> String {
        let p = rl_last_error();
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
}
