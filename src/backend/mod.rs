//! wgpu backend: device + queue + pipeline cache + buffer allocator.

mod context;
pub mod elementwise;
pub mod matmul;
mod spike;

pub use context::WgpuCtx;
pub use spike::compute_spike;
