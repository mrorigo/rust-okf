use crate::query::{QueryPlan, SearchResult};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchModeRequest {
    Lexical,
    Vector,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub mode: Option<SearchModeRequest>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub plan: QueryPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DocumentInput {
    pub bundle_path: String,
    pub file_path: String,
    pub frontmatter: serde_json::Map<String, serde_json::Value>,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteRequest {
    #[serde(default)]
    pub doc_ids: Vec<String>,
    #[serde(default)]
    pub logical_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
}
