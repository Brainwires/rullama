// MCP module - Model Context Protocol client and tool integration
//
// Re-exports from the rullama-mcp-client framework crate, plus CLI-specific tool adapter.

pub use rullama::mcp::*;

// CLI-specific MCP tool adapter
mod tool_adapter;
pub use tool_adapter::*;
