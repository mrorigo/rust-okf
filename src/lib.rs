pub mod api;
/// Rust guideline compliant 2026-06-17
///
/// Public library entry points for the OKF indexing engine.
pub mod bm25;
pub mod config;
pub mod embedding;
pub mod index;
pub mod okf;
pub mod openapi;
pub mod query;
pub mod schema;
pub mod storage;

pub use api::serve as serve_http;
pub use config::{AppConfig, FastEmbedConfig};
pub use embedding::FastEmbedProvider;
pub use embedding::{EmbeddingProvider, MockEmbeddingProvider};
pub use index::{open_index, Index, IndexConfig, IndexError, SearchMode};
pub use okf::{load_bundle, OkfDocument, OkfDocumentBuilder};
pub use openapi::ApiDoc;
