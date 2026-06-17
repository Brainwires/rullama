//! rullama-core — C-ABI FFI shim over the published `rullama` crate.
//!
//! ## The dominant constraint: the native `Model` is `!Send`
//!
//! `rullama`'s `TensorFetcher` is `#[async_trait(?Send)]` and the `Model` +
//! wgpu futures are intentionally single-threaded. A `Model` therefore cannot
//! be moved between threads. The pattern this crate establishes:
//!
//! * Each engine handle (`RlModel`) owns **one** OS thread for its entire
//!   lifetime. The `Model` (and the `WgpuCtx`) are created on that thread and
//!   never leave it.
//! * Every C-ABI call marshals a [`Command`] to that thread over an MPSC
//!   channel and blocks for the reply (streaming generation will instead push
//!   tokens through a callback — added in M1).
//! * Async engine methods are driven with `pollster::block_on` **on the owning
//!   thread** — the same approach proven in the upstream `tools/ios-bench`.
//!
//! Only `Send` values cross the channel (owned `String`/`Vec<u8>`/scalars);
//! the `!Send` engine state stays put.
//!
//! M0 scope: handle lifecycle + a wgpu probe that exercises the full
//! owning-thread + `block_on` + wgpu-init path on the real GPU. Model load,
//! tokenize, and streaming generation land in M1.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;

use rullama::backend::WgpuCtx;

// ---------------------------------------------------------------------------
// Thread-local last-error (set on the calling thread, read on the same thread)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let c = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Returns the last error message for the **calling thread** as a NUL-terminated
/// C string, or NULL if none. The pointer is valid until the next FFI call on
/// this thread; copy it immediately. Do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn rl_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr())
    })
}

/// Returns this shim's version string (static; do not free).
#[unsafe(no_mangle)]
pub extern "C" fn rl_version() -> *const c_char {
    // Static, NUL-terminated.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Frees a C string previously returned by this library (e.g. `rl_model_probe`).
///
/// # Safety
/// `ptr` must be NULL or a pointer obtained from this library; it must not be
/// used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_free_str(ptr: *mut c_char) {
    if !ptr.is_null() {
        // Reclaim ownership and drop.
        drop(unsafe { CString::from_raw(ptr) });
    }
}

fn into_c_string(s: impl Into<Vec<u8>>) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("<nul-in-string>").unwrap())
        .into_raw()
}

// ---------------------------------------------------------------------------
// Owning-thread command loop
// ---------------------------------------------------------------------------

/// Commands marshalled to a model handle's owning thread. Replies travel back
/// over a per-command channel carrying only `Send` values.
enum Command {
    /// Probe the GPU: lazily init `WgpuCtx` on the owning thread and report
    /// adapter info as a human-readable string.
    Probe(Sender<Result<String, String>>),
    /// Tear down the worker loop.
    Shutdown,
}

/// The worker loop: owns the `!Send` engine state (here: the `WgpuCtx`) and
/// services commands until shutdown. Runs `pollster::block_on` for async work.
fn worker(rx: mpsc::Receiver<Command>) {
    // The wgpu context is created lazily on first probe and then reused. It is
    // `!Send` and lives only on this thread — exactly the invariant we need.
    let mut ctx: Option<WgpuCtx> = None;

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
                let c = ctx.as_ref().expect("ctx initialized above");
                let info = c.adapter.get_info();
                let limits = c.adapter.limits();
                let desc = format!(
                    "adapter={} backend={:?} device_type={:?} subgroups={} has_f16={} max_storage_buffer_binding_size={}",
                    info.name,
                    info.backend,
                    info.device_type,
                    c.has_subgroups,
                    c.has_f16,
                    limits.max_storage_buffer_binding_size,
                );
                let _ = reply.send(Ok(desc));
            }
            Command::Shutdown => break,
        }
    }
    // `ctx` drops here on the owning thread — the only thread that may drop it.
}

/// Opaque engine handle. Owns the worker thread + the channel to it.
pub struct RlModel {
    tx: Sender<Command>,
    handle: Option<JoinHandle<()>>,
}

/// Creates an engine handle: spawns its dedicated owning thread.
///
/// Returns NULL if the thread could not be spawned (see `rl_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn rl_model_create() -> *mut RlModel {
    let (tx, rx) = mpsc::channel();
    match std::thread::Builder::new()
        .name("rullama-model".into())
        .spawn(move || worker(rx))
    {
        Ok(handle) => Box::into_raw(Box::new(RlModel {
            tx,
            handle: Some(handle),
        })),
        Err(e) => {
            set_last_error(format!("failed to spawn model thread: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Frees an engine handle: signals the owning thread to stop and joins it,
/// guaranteeing the `!Send` engine state is dropped on its own thread.
///
/// # Safety
/// `m` must be NULL or a handle from `rl_model_create`, not used afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_model_free(m: *mut RlModel) {
    if m.is_null() {
        return;
    }
    let mut model = unsafe { Box::from_raw(m) };
    let _ = model.tx.send(Command::Shutdown);
    if let Some(h) = model.handle.take() {
        let _ = h.join();
    }
}

/// Probes the GPU on the handle's owning thread. On success writes a newly
/// allocated C string (free with `rl_free_str`) describing the adapter to
/// `*out` and returns 0. Negative on error (see `rl_last_error`).
///
/// # Safety
/// `m` must be a valid handle; `out` must be a valid, writable `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_model_probe(m: *mut RlModel, out: *mut *mut c_char) -> i32 {
    let Some(model) = (unsafe { m.as_ref() }) else {
        set_last_error("rl_model_probe: null handle");
        return -1;
    };
    if out.is_null() {
        set_last_error("rl_model_probe: null out pointer");
        return -2;
    }
    let (rtx, rrx) = mpsc::channel();
    if model.tx.send(Command::Probe(rtx)).is_err() {
        set_last_error("rl_model_probe: worker thread is gone");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(desc)) => {
            unsafe { *out = into_c_string(desc) };
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("rl_model_probe: worker dropped the reply");
            -5
        }
    }
}

/// Convenience one-shot: create a handle, probe, free. On success writes a
/// newly allocated C string (free with `rl_free_str`) to `*out`, returns 0.
///
/// # Safety
/// `out` must be a valid, writable `*mut c_char`.
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

// ---------------------------------------------------------------------------
// Native smoke test — exercises owning-thread + block_on + real wgpu init.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn probe_reports_an_adapter() {
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { rl_wgpu_probe(&mut out) };
        assert_eq!(rc, 0, "probe failed: rc={rc}");
        assert!(!out.is_null());
        let desc = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
        eprintln!("probe: {desc}");
        assert!(desc.contains("adapter="), "unexpected probe output: {desc}");
        unsafe { rl_free_str(out) };
    }

    #[test]
    fn create_free_roundtrip_joins_thread() {
        let m = rl_model_create();
        assert!(!m.is_null());
        unsafe { rl_model_free(m) };
    }
}
