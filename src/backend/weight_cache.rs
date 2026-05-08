//! Lazy GPU weight buffer cache.
//!
//! Each tensor in the GGUF gets uploaded to a `wgpu::Buffer` on first access; future
//! calls return clones of the same buffer (wgpu Buffers are Arc-internally, so
//! cloning is `Arc::clone` and free). Eliminates the per-call weight upload that
//! dominates `forward_token_gpu` cost.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, RullamaError};
use crate::gguf::{GgmlDtype, GgufReader};

/// One tile of a row-tiled tensor.
pub struct TiledTensor {
    pub buffer: wgpu::Buffer,
    /// Index of the first row (along the slow / second axis) covered by this buffer.
    pub row_start: usize,
    /// Number of rows covered.
    pub n_rows: usize,
}

pub struct WeightCache {
    reader: Arc<GgufReader>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: RefCell<HashMap<String, wgpu::Buffer>>,
    tiles: RefCell<HashMap<(String, usize), Vec<wgpu::Buffer>>>,
    tile_meta: RefCell<HashMap<(String, usize), Vec<(usize, usize)>>>,
}

impl WeightCache {
    pub fn new(reader: Arc<GgufReader>, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            reader,
            device,
            queue,
            buffers: RefCell::new(HashMap::new()),
            tiles: RefCell::new(HashMap::new()),
            tile_meta: RefCell::new(HashMap::new()),
        }
    }

    /// Borrow of the underlying GGUF reader (for callers that occasionally need an
    /// f32 dequant outside the GPU buffer path — e.g. the small RoPE freq-factors tensor).
    pub fn reader(&self) -> &GgufReader { &self.reader }

    /// Get the GPU buffer for the named tensor, uploading on first access.
    pub fn buffer(&self, name: &str) -> Result<wgpu::Buffer> {
        if let Some(b) = self.buffers.borrow().get(name) {
            return Ok(b.clone());
        }
        let bytes = self.reader.tensor_bytes(name)?;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        let cloned = buf.clone();
        self.buffers.borrow_mut().insert(name.to_string(), buf);
        Ok(cloned)
    }

    /// Best-effort buffer fetch: Ok(None) if the tensor is absent.
    pub fn buffer_opt(&self, name: &str) -> Result<Option<wgpu::Buffer>> {
        if self.reader.tensor(name).is_err() {
            return Ok(None);
        }
        self.buffer(name).map(Some)
    }

    /// Look up a tensor's GGML dtype (without uploading).
    pub fn dtype(&self, name: &str) -> Result<GgmlDtype> {
        Ok(self.reader.tensor(name)?.dtype)
    }

    pub fn cached_count(&self) -> usize {
        self.buffers.borrow().len()
    }

    pub fn cached_bytes(&self) -> u64 {
        let single: u64 = self.buffers.borrow().values().map(|b| b.size()).sum();
        let tiled: u64 = self.tiles.borrow().values()
            .flat_map(|v| v.iter().map(|b| b.size()))
            .sum();
        single + tiled
    }

    /// Split a 2-D quantized tensor along its slow (second) axis into multiple GPU
    /// buffers, each ≤ `max_bytes_per_tile` bytes.
    pub fn buffer_tiles(&self, name: &str, max_bytes_per_tile: usize) -> Result<Vec<TiledTensor>> {
        let key = (name.to_string(), max_bytes_per_tile);
        {
            let tiles = self.tiles.borrow();
            let meta = self.tile_meta.borrow();
            if let (Some(bufs), Some(metas)) = (tiles.get(&key), meta.get(&key)) {
                return Ok(bufs.iter().zip(metas.iter())
                    .map(|(buf, &(row_start, n_rows))| TiledTensor {
                        buffer: buf.clone(), row_start, n_rows
                    })
                    .collect());
            }
        }

        let desc = self.reader.tensor(name)?;
        if desc.dims.len() != 2 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: tensor {name} has {} dims, expected 2", desc.dims.len()
            )));
        }
        let row_len = desc.dims[0] as usize;
        let n_rows = desc.dims[1] as usize;
        let block_elems = desc.dtype.block_elems();
        if row_len % block_elems != 0 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: row_len {row_len} not multiple of block_elems {block_elems}"
            )));
        }
        let blocks_per_row = row_len / block_elems;
        let row_bytes = blocks_per_row * desc.dtype.block_bytes();
        if row_bytes == 0 {
            return Err(RullamaError::Inference(format!(
                "buffer_tiles: row_bytes is 0 for {name}"
            )));
        }
        let rows_per_tile = (max_bytes_per_tile / row_bytes).max(1);
        let all_bytes = self.reader.tensor_bytes(name)?;

        let mut bufs = Vec::new();
        let mut metas = Vec::new();
        let mut row_start = 0usize;
        while row_start < n_rows {
            let row_end = (row_start + rows_per_tile).min(n_rows);
            let byte_start = row_start * row_bytes;
            let byte_end   = row_end   * row_bytes;
            let chunk = &all_bytes[byte_start..byte_end];
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{name}#tile{row_start}")),
                size: chunk.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, chunk);
            metas.push((row_start, row_end - row_start));
            bufs.push(buf);
            row_start = row_end;
        }

        let result: Vec<TiledTensor> = bufs.iter().zip(metas.iter())
            .map(|(buf, &(rs, nr))| TiledTensor { buffer: buf.clone(), row_start: rs, n_rows: nr })
            .collect();
        self.tiles.borrow_mut().insert(key.clone(), bufs);
        self.tile_meta.borrow_mut().insert(key, metas);
        Ok(result)
    }
}
