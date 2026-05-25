//! Bind-group cache for LoRA dispatchers.
//!
//! Background: every LoRA dispatch on the inference path currently
//! creates a fresh `wgpu::BindGroup` (and a fresh uniform `wgpu::Buffer`
//! to hold its `(scale, k, n, accumulate, …)` params). On browser
//! WebGPU each `create_bind_group` is ~50-150 µs of overhead — and
//! the adapter path fires ~500 LoRA dispatches per generated token.
//! That's ~50 ms/tok lost to bind-group allocation alone, while the
//! underlying buffers and pipeline never change between tokens.
//!
//! Design:
//!
//! - Cache key is `(pipeline_id, storage_buffer_ids…)` — the
//!   identity of the GPU resources the bind group actually binds.
//!   These are stable for the lifetime of a loaded adapter
//!   (`InferenceLoraLayer` owns the A/B/z buffers and only
//!   `loadAdapter`/`clearAdapter` swap them).
//! - Each cache entry OWNS a dedicated uniform buffer. On cache
//!   hit, the dispatcher calls `queue.write_buffer` to update the
//!   uniform with this call's params, then reuses the cached bind
//!   group. wgpu sequences `write_buffer` before any dispatch in
//!   the same submission, so the dispatch reads the freshly-written
//!   params.
//! - Mutex (not RefCell) because wgpu handles are `Send + Sync`
//!   and the cache lives inside `WgpuCtx` which is `Clone` and
//!   shared across cloned ctx handles. The lock is held only for
//!   the HashMap lookup/insert — microseconds of contention.
//! - Cleared by `clear()` on `loadAdapter` / `clearAdapter`. The
//!   old cache entries' buffer handles drop, freeing GPU memory
//!   (uniform buffers are 32 bytes each; ~500 entries = 16 KB —
//!   the leak even if we forget to clear is trivial).

use std::collections::HashMap;
use std::sync::Mutex;

/// Identifier for a wgpu resource — wgpu doesn't expose a stable
/// `Id` type publicly so we hash the resource's pointer address.
/// Stable for the resource's lifetime. The cache is invalidated
/// when adapter buffers are dropped (via `clear()`), so a freed-
/// then-recycled pointer can never produce a false-positive hit.
fn buf_id(b: &wgpu::Buffer) -> u64 {
    b as *const wgpu::Buffer as usize as u64
}
fn pipeline_id(p: &wgpu::ComputePipeline) -> u64 {
    p as *const wgpu::ComputePipeline as usize as u64
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub pipeline: u64,
    pub b0: u64,
    pub b1: u64,
    pub b2: u64,
    /// `Some(_)` for kernels with four storage bindings
    /// (e.g. `lora_outer_add` binds two inputs + one output, plus
    /// the uniform; future fused kernels may bind more).
    pub b3: Option<u64>,
}

impl CacheKey {
    pub fn two(p: &wgpu::ComputePipeline, b0: &wgpu::Buffer, b1: &wgpu::Buffer) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: 0,
            b3: None,
        }
    }
    pub fn three(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: None,
        }
    }
    pub fn four(
        p: &wgpu::ComputePipeline,
        b0: &wgpu::Buffer,
        b1: &wgpu::Buffer,
        b2: &wgpu::Buffer,
        b3: &wgpu::Buffer,
    ) -> Self {
        Self {
            pipeline: pipeline_id(p),
            b0: buf_id(b0),
            b1: buf_id(b1),
            b2: buf_id(b2),
            b3: Some(buf_id(b3)),
        }
    }
}

#[derive(Clone)]
pub struct CachedDispatch {
    /// Persistent uniform buffer owned by this cache entry. The
    /// caller writes the per-call params here BEFORE dispatching,
    /// then submits — wgpu sequences the write before the
    /// dispatch on the same queue.
    pub uniform: wgpu::Buffer,
    /// Cached bind group; references `uniform` plus the storage
    /// buffers the key was derived from.
    pub bind_group: wgpu::BindGroup,
}

pub struct LoraBindGroupCache {
    inner: Mutex<HashMap<CacheKey, CachedDispatch>>,
}

impl LoraBindGroupCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Get or insert a cache entry. `build` is called only on miss;
    /// it must construct both the uniform buffer and the bind group
    /// that references it. The returned handles are clones (wgpu
    /// types are Arc-internal — cheap).
    pub fn get_or_create<F>(&self, key: CacheKey, build: F) -> CachedDispatch
    where
        F: FnOnce() -> CachedDispatch,
    {
        let mut guard = self.inner.lock().unwrap();
        guard.entry(key).or_insert_with(build).clone()
    }

    /// Drop every cached entry. Called when the adapter is loaded
    /// or cleared — the buffer ids in the keys would otherwise be
    /// stale (and could collide with freshly-allocated buffers at
    /// the same address).
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

impl Default for LoraBindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}
