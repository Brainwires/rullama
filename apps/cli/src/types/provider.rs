//! Provider types.
//!
//! Re-exports ChatOptions from rullama-core and ProviderType/ProviderConfig from rullama-provider.

// Re-export ChatOptions from framework
pub use rullama::core::provider::ChatOptions;

// Re-export from providers crate
pub use rullama::providers::ProviderConfig;
pub use rullama::providers::ProviderType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_options_default() {
        let opts = ChatOptions::default();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.max_tokens, Some(4096));
        assert!(opts.system.is_none());
        assert!(opts.top_p.is_none());
    }

    #[test]
    fn test_provider_type_default_model() {
        assert_eq!(ProviderType::Anthropic.default_model(), "claude-sonnet-4-6");
    }

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!(
            ProviderType::from_str_opt("anthropic"),
            Some(ProviderType::Anthropic)
        );
        assert_eq!(
            ProviderType::from_str_opt("openai"),
            Some(ProviderType::OpenAI)
        );
        assert_eq!(
            ProviderType::from_str_opt("google"),
            Some(ProviderType::Google)
        );
        assert_eq!(
            ProviderType::from_str_opt("ollama"),
            Some(ProviderType::Ollama)
        );
    }

    #[test]
    fn test_provider_type_as_str() {
        assert_eq!(ProviderType::Anthropic.as_str(), "anthropic");
        assert_eq!(ProviderType::OpenAI.as_str(), "openai");
    }
}
