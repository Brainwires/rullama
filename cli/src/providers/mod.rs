// Re-export provider trait from core
pub use rullama::core::provider::Provider;

// Re-export specific items from framework providers crate (not glob, to avoid ProviderFactory collision)
pub use rullama::providers::{
    // Model listing
    AvailableModel,
    ModelCapability,
    ModelLister,
    OllamaProvider,
    ProviderConfig,
    ProviderType,
    RateLimitedClient,
    RateLimiter,
    create_model_lister,
};

// Re-export sub-modules for `use crate::providers::local_llm::Foo` patterns
pub mod local_llm {
    pub use rullama::providers::local_llm::*;
}

// Chat provider factory (canonical factory for creating providers)
pub use rullama::providers::ChatProviderFactory;

// CLI-specific: factory depends on SessionManager/AuthClient
mod factory;
pub use factory::*;
