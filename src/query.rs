/// Rust guideline compliant 2026-06-17
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

/// Query execution trace.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueryPlan {
    /// Original query text.
    pub query: String,
    /// Lexical candidate list.
    pub lexical_candidates: Vec<(String, f32)>,
    /// Dense candidate list.
    pub vector_candidates: Vec<(String, f32)>,
    /// Fused ranking.
    pub fused: Vec<(String, f32)>,
}

/// Search hit payload returned by the engine.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchResult {
    pub doc_id: String,
    pub concept_path: String,
    pub title: Option<String>,
    pub type_name: String,
    pub tags: Vec<String>,
    pub snippet: String,
    pub bm25_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub fused_score: f32,
}

/// Fuses lexical and vector rankings with Reciprocal Rank Fusion.
pub fn rrf_fuse(lexical: &[(String, f32)], vector: &[(String, f32)], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (rank, (doc_id, _)) in lexical.iter().enumerate() {
        *scores.entry(doc_id.clone()).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, (doc_id, _)) in vector.iter().enumerate() {
        *scores.entry(doc_id.clone()).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}
