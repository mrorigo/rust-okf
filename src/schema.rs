/// Rust guideline compliant 2026-06-17
use crate::query::{QueryPlan, SearchResult};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Search mode requested by API clients.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchModeRequest {
    Lexical,
    Vector,
    Hybrid,
}

pub use crate::query::QueryFilter;

/// Search request payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub mode: Option<SearchModeRequest>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub filter: Option<QueryFilter>,
}

/// Search response payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub plan: QueryPlan,
    pub total_hits: usize,
    pub offset: usize,
    pub top_k: usize,
}

/// Document ingestion payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DocumentInput {
    pub bundle_path: String,
    pub file_path: String,
    pub frontmatter: serde_json::Map<String, serde_json::Value>,
    pub body: String,
}

/// Delete request payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteRequest {
    #[serde(default)]
    pub doc_ids: Vec<String>,
    #[serde(default)]
    pub logical_keys: Vec<String>,
}

/// Simple status response payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
}
