//! GGUF v3 parser.
//!
//! Browser-friendly: takes `&[u8]` (a `Uint8Array` slice on wasm32), no mmap, no I/O.
//! Hand-rolled rather than depending on a crate so we own the wasm story end-to-end and
//! the dep tree stays small.
//!
//! Spec reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

mod dtype;
mod reader;
mod value;

pub mod tensor;
pub mod quant;

pub use dtype::GgmlDtype;
pub use reader::{GgufReader, TensorDesc};
pub use tensor::{dequant_row_to_f32, dequant_tensor_to_f32};
pub use value::{GgufValue, GgufValueType};
