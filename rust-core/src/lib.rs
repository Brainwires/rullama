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

use rullama::api::Model;
use rullama::backend::WgpuCtx;
use rullama::embed::EmbeddingModel;
use rullama::gguf::{FileFetcher, TensorFetcher};
use rullama::sampling::SamplingOptions;
use rullama::tts::KokoroTts;

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
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| "invalid utf-8")
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
    Shutdown,
}

fn worker(rx: mpsc::Receiver<Command>, cancel: Arc<AtomicBool>) {
    // `!Send` engine state — created and dropped only on this thread.
    let mut ctx: Option<WgpuCtx> = None;
    let mut model: Option<Model> = None;

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
            Command::LoadPath { path, max_ctx, text_only, reply } => {
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
                let _ = reply.send(match model.as_ref() {
                    Some(m) => Ok(m.encode_tokens(&text)),
                    None => Err("no model loaded".into()),
                });
            }
            Command::TokenStr { id, reply } => {
                let _ = reply.send(model.as_ref().and_then(|m| m.token_str_native(id)));
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
                let _ = reply.send(model.as_ref().map_or(0, |m| m.vocab_size_native()));
            }
            Command::Position(reply) => {
                let _ = reply.send(model.as_ref().map_or(0, |m| m.position_native()));
            }
            Command::Generate { prompt, max_new, cb, reply } => {
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
                    if audio { m.audio_sentinel_ids_native() } else { m.image_sentinel_ids_native() }
                });
                let _ = reply.send(r);
            }
            Command::ImageSoftCount { h, w, reply } => {
                let _ = reply.send(model.as_ref().and_then(|m| m.image_soft_token_count_native(h, w)));
            }
            Command::EncodeImage { pixels, h, w, reply } => {
                let res = match model.as_mut() {
                    Some(m) => pollster::block_on(m.encode_image_native(&pixels, h, w, None))
                        .map_err(|e| format!("{e}")),
                    None => Err("no model loaded".into()),
                };
                let _ = reply.send(res);
            }
            Command::EncodeAudio { pcm, reply } => {
                let res = match model.as_mut() {
                    Some(m) => pollster::block_on(m.encode_audio_native(&pcm)).map_err(|e| format!("{e}")),
                    None => Err("no model loaded".into()),
                };
                let _ = reply.send(res);
            }
            Command::GenerateSpliced {
                prompt, sentinel_begin, soft, d_text, max_new, cb, reply,
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
        let piece = m.token_str_native(cur).unwrap_or_default().replace('\u{2581}', " ");
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
pub unsafe extern "C" fn rl_image_sentinel_ids(m: *mut RlModel, begin: *mut u32, end: *mut u32) -> i32 {
    unsafe { sentinels(m, false, begin, end) }
}

/// Audio sentinel token ids (begin, end). Returns -7 if absent.
/// # Safety
/// `m` valid; `begin`/`end` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_audio_sentinel_ids(m: *mut RlModel, begin: *mut u32, end: *mut u32) -> i32 {
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
    let pixels = if pixels.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(pixels, n) }.to_vec() };
    match call(m, |reply| Command::EncodeImage { pixels, h, w, reply }) {
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
    let pcm = if pcm.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(pcm, n) }.to_vec() };
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
pub unsafe extern "C" fn rl_decode_wav(bytes: *const u8, n: usize, out: *mut *mut f32, out_len: *mut usize) -> i32 {
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
    let prompt = if prompt.is_null() || n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(prompt, n) }.to_vec() };
    let soft = if soft.is_null() || soft_len == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(soft, soft_len) }.to_vec() };
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
    Load { bytes: Vec<u8>, reply: Sender<Result<(), String>> },
    SetLexicon { gold: Vec<u8>, silver: Vec<u8>, reply: Sender<()> },
    Synthesize { text: String, voice: String, reply: Sender<Result<Vec<f32>, String>> },
    Shutdown,
}

fn tts_worker(rx: mpsc::Receiver<TtsCommand>) {
    let mut tts: Option<KokoroTts> = None;
    for cmd in rx {
        match cmd {
            TtsCommand::Load { bytes, reply } => {
                let r = pollster::block_on(KokoroTts::load_native(bytes)).map_err(|e| format!("{e}"));
                match r {
                    Ok(t) => { tts = Some(t); let _ = reply.send(Ok(())); }
                    Err(e) => { let _ = reply.send(Err(e)); }
                }
            }
            TtsCommand::SetLexicon { gold, silver, reply } => {
                if let Some(t) = tts.as_mut() { t.set_lexicon_native(&gold, &silver); }
                let _ = reply.send(());
            }
            TtsCommand::Synthesize { text, voice, reply } => {
                let r = match tts.as_mut() {
                    Some(t) => {
                        let (pcm, _oov) = pollster::block_on(t.synthesize_native(&text, &voice, None));
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
    match std::thread::Builder::new().name("rullama-tts".into()).spawn(move || tts_worker(rx)) {
        Ok(handle) => Box::into_raw(Box::new(RlTts { tx, handle: Some(handle) })),
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
    if t.is_null() { return; }
    let mut b = unsafe { Box::from_raw(t) };
    let _ = b.tx.send(TtsCommand::Shutdown);
    if let Some(h) = b.handle.take() { let _ = h.join(); }
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
    rx.recv().map_err(|_| { set_last_error("tts dropped reply"); -3 })
}

/// Load the Kokoro GGUF from a path.
/// # Safety
/// `t` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_load_path(t: *mut RlTts, path: *const c_char) -> i32 {
    let path = match unsafe { c_str(path) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    let bytes = match std::fs::read(&path) { Ok(b) => b, Err(e) => { set_last_error(format!("read {path}: {e}")); return -6; } };
    match tts_call(t, |reply| TtsCommand::Load { bytes, reply }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => { set_last_error(e); -6 }
        Err(c) => c,
    }
}

/// Set the G2P lexicon from gold (+ optional silver) JSON file paths.
/// # Safety
/// `t` valid; paths valid C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_set_lexicon(t: *mut RlTts, gold_path: *const c_char, silver_path: *const c_char) -> i32 {
    let gp = match unsafe { c_str(gold_path) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    let sp = match unsafe { c_str(silver_path) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    let gold = std::fs::read(&gp).unwrap_or_default();
    let silver = std::fs::read(&sp).unwrap_or_default();
    match tts_call(t, |reply| TtsCommand::SetLexicon { gold, silver, reply }) {
        Ok(()) => 0,
        Err(c) => c,
    }
}

/// Synthesize text → mono f32 PCM at `rl_tts_sample_rate` (free with `rl_free_f32`).
/// # Safety
/// `t` valid; `text`/`voice` valid C strings; `out`/`out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_tts_synthesize(t: *mut RlTts, text: *const c_char, voice: *const c_char, out: *mut *mut f32, out_len: *mut usize) -> i32 {
    if out.is_null() || out_len.is_null() { set_last_error("null out"); return -2; }
    let text = match unsafe { c_str(text) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    let voice = match unsafe { c_str(voice) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    match tts_call(t, |reply| TtsCommand::Synthesize { text, voice, reply }) {
        Ok(Ok(pcm)) => { put_f32(pcm, out, out_len); 0 }
        Ok(Err(e)) => { set_last_error(e); -6 }
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
// Embeddings (EmbeddingGemma) — a SEPARATE model handle with its own thread.
// ---------------------------------------------------------------------------

enum EmbedCommand {
    Load { bytes: Vec<u8>, reply: Sender<Result<(), String>> },
    Dim(Sender<u32>),
    Embed { text: String, target_dim: usize, reply: Sender<Result<Vec<f32>, String>> },
    Shutdown,
}

fn embed_worker(rx: mpsc::Receiver<EmbedCommand>) {
    let mut model: Option<EmbeddingModel> = None;
    for cmd in rx {
        match cmd {
            EmbedCommand::Load { bytes, reply } => {
                let r = pollster::block_on(EmbeddingModel::load_native(bytes)).map_err(|e| format!("{e}"));
                match r {
                    Ok(m) => { model = Some(m); let _ = reply.send(Ok(())); }
                    Err(e) => { let _ = reply.send(Err(e)); }
                }
            }
            EmbedCommand::Dim(reply) => {
                let _ = reply.send(model.as_ref().map_or(0, |m| m.dim_native()));
            }
            EmbedCommand::Embed { text, target_dim, reply } => {
                let r = match model.as_ref() {
                    Some(m) => pollster::block_on(m.embed_native(&text, target_dim)).map_err(|e| format!("{e}")),
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
    match std::thread::Builder::new().name("rullama-embed".into()).spawn(move || embed_worker(rx)) {
        Ok(handle) => Box::into_raw(Box::new(RlEmbed { tx, handle: Some(handle) })),
        Err(e) => { set_last_error(format!("failed to spawn embed thread: {e}")); std::ptr::null_mut() }
    }
}

/// # Safety
/// `t` must be NULL or a handle from `rl_embed_create`, unused afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed_free(t: *mut RlEmbed) {
    if t.is_null() { return; }
    let mut b = unsafe { Box::from_raw(t) };
    let _ = b.tx.send(EmbedCommand::Shutdown);
    if let Some(h) = b.handle.take() { let _ = h.join(); }
}

fn embed_call<T>(t: *mut RlEmbed, make: impl FnOnce(Sender<T>) -> EmbedCommand) -> Result<T, i32> {
    let Some(e) = (unsafe { t.as_ref() }) else { set_last_error("null embed handle"); return Err(-1); };
    let (tx, rx) = mpsc::channel();
    if e.tx.send(make(tx)).is_err() { set_last_error("embed worker gone"); return Err(-2); }
    rx.recv().map_err(|_| { set_last_error("embed dropped reply"); -3 })
}

/// Load the embedding GGUF from a path.
/// # Safety
/// `t` valid; `path` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_embed_load_path(t: *mut RlEmbed, path: *const c_char) -> i32 {
    let path = match unsafe { c_str(path) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    let bytes = match std::fs::read(&path) { Ok(b) => b, Err(e) => { set_last_error(format!("read {path}: {e}")); return -6; } };
    match embed_call(t, |reply| EmbedCommand::Load { bytes, reply }) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => { set_last_error(e); -6 }
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
pub unsafe extern "C" fn rl_embed(t: *mut RlEmbed, text: *const c_char, target_dim: usize, out: *mut *mut f32, out_len: *mut usize) -> i32 {
    if out.is_null() || out_len.is_null() { set_last_error("null out"); return -2; }
    let text = match unsafe { c_str(text) } { Ok(s) => s.to_owned(), Err(e) => { set_last_error(e); return -5; } };
    match embed_call(t, |reply| EmbedCommand::Embed { text, target_dim, reply }) {
        Ok(Ok(v)) => { put_f32(v, out, out_len); 0 }
        Ok(Err(e)) => { set_last_error(e); -6 }
        Err(c) => c,
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
        let desc = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
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

        let prompt = "<start_of_turn>user\nSay hello in one word.<end_of_turn>\n<start_of_turn>model\n";
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
                unsafe { CStr::from_ptr(piece) }.to_string_lossy().into_owned()
            };
            eprintln!("tok #{count}: {tok} = {s:?}");
        }
        let mut count: u32 = 0;
        let produced = unsafe {
            rl_generate(m, ids, nn, 16, on_tok, &mut count as *mut u32 as *mut c_void)
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
