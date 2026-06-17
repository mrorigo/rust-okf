pub mod bm25;
pub mod config;
pub mod api;
pub mod embedding;
pub mod index;
pub mod openapi;
pub mod okf;
pub mod schema;
pub mod query;
pub mod storage;

pub use embedding::{EmbeddingProvider, MockEmbeddingProvider};
pub use embedding::FastEmbedProvider;
pub use config::{AppConfig, FastEmbedConfig};
pub use api::serve as serve_http;
pub use openapi::ApiDoc;
pub use index::{open_index, Index, IndexConfig, IndexError, SearchMode};
pub use okf::{load_bundle, OkfDocument, OkfDocumentBuilder};
