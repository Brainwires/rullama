// Storage namespace — internal CLI alias.
//
// The actual store implementations live in the framework crates after Phase 10:
//
//   rullama-storage  — primitives: StorageBackend, LanceDatabase, embeddings, BM25
//   rullama-stores   — opinionated minimum store set: session, task, plan,
//                         conversation, template, lock, image + tiered memory
//   rullama-agent    — SEAL-domain stores (pattern_store + LanceDatabaseExt)
//
// This module is a re-export aggregator only — kept so the 29 CLI files that
// already use `crate::storage::{...}` don't need to be rewritten. New code
// should import directly from `rullama_stores::` / `rullama_storage::`.

pub use rullama::storage::databases::VectorDatabase;
pub use rullama::storage::*;

// FileContextManager + FileContent + FileChunk moved to rullama-core in Phase 9.
pub use rullama_core::file_context::{FileChunk, FileContent, FileContextManager};

// The opinionated store set (sessions, tasks, plans, conversations, locks,
// images, templates) + the tier schema stores (Message/Summary/Fact/MentalModel/
// TierMetadata + tier_types).
pub use rullama_stores::*;

// Tiered memory orchestration (TieredMemory, MultiFactorScore, …) + offline
// dream consolidation. The schema stores live in rullama-stores; this
// crate adds the engines that operate on them.
pub use rullama_memory::{
    CanonicalWriteToken, MultiFactorScore, TieredMemory, TieredMemoryConfig, TieredMemoryStats,
    TieredSearchResult,
};

// Document types (live in rullama-rag::rag::documents)
pub use rullama_rag::rag::documents::{
    ChunkerConfig, DocumentBM25Manager, DocumentChunk, DocumentChunker, DocumentMetadata,
    DocumentMetadataStore, DocumentProcessor, DocumentScope, DocumentSearchRequest,
    DocumentSearchResult, DocumentStore, DocumentType, ExtractedDocument,
    lance_tables as document_lance_tables,
};

// CLI-domain helpers that stayed behind:
//   - PlanModeStore couples to CLI message/plan-mode types.
//   - PersistentTaskManager wraps rullama-agent's TaskManager and has
//     zero in-tree consumers; kept as a CLI-local primitive for now.
pub use crate::persistent_task_manager::PersistentTaskManager;
pub use crate::plan_mode_store::PlanModeStore;

// SEAL pattern store moved to rullama-agent::seal where its types live.
pub use rullama_seal::pattern_store::{LanceDatabaseExt, PatternMetadata, PatternStore};
