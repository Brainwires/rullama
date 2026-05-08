//! Lazy weight access for the CPU reference forward pass.
//!
//! Each `load(name)` call reads + dequantizes the named tensor into a fresh `Vec<f32>`.
//! No caching: callers drop weights when they're done with a layer to keep peak memory
//! bounded. This is fine for parity testing where wall-clock doesn't matter.

use crate::error::Result;
use crate::gguf::{GgufReader, dequant_row_to_f32, dequant_tensor_to_f32};

/// Borrow-only handle that knows how to materialize tensors on demand.
pub struct Weights<'a> {
    reader: &'a GgufReader<'a>,
}

impl<'a> Weights<'a> {
    pub fn new(reader: &'a GgufReader<'a>) -> Self {
        Self { reader }
    }

    pub fn reader(&self) -> &GgufReader<'a> { self.reader }

    /// Load and dequantize a whole tensor into f32. Allocates.
    pub fn load(&self, name: &str) -> Result<Vec<f32>> {
        dequant_tensor_to_f32(self.reader, name)
    }

    /// Load and dequantize a single row of a 2-D tensor into f32. Useful for embedding
    /// lookups on huge tables (token_embd, per_layer_token_embd).
    pub fn load_row(&self, name: &str, row_idx: usize) -> Result<Vec<f32>> {
        dequant_row_to_f32(self.reader, name, row_idx)
    }

    /// Best-effort load: returns Ok(None) if the tensor isn't present (some optional
    /// tensors like rope_freqs.weight may be absent on smaller variants).
    pub fn load_opt(&self, name: &str) -> Result<Option<Vec<f32>>> {
        match self.reader.tensor(name) {
            Ok(_) => self.load(name).map(Some),
            Err(_) => Ok(None),
        }
    }
}
