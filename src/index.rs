/// Rust guideline compliant 2026-06-17
use crate::bm25::{Bm25Config, Bm25Index};
use crate::embedding::EmbeddingProvider;
use crate::okf::OkfDocument;
use crate::query::{rrf_fuse, QueryFilter, QueryPlan, SearchResult};
use crate::storage::{
    results_from_docs, IndexStorage, Manifest, SegmentFile, SegmentMetadata, SegmentView,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Index-level tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// BM25 configuration.
    pub bm25: Bm25Config,
    /// Reciprocal Rank Fusion constant.
    pub rrf_k: f32,
    /// HNSW ANN configuration.
    pub ann: crate::ann::AnnConfig,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            bm25: Bm25Config::default(),
            rrf_k: 60.0,
            ann: crate::ann::AnnConfig::default(),
        }
    }
}

/// Search mode for the public API.
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Lexical,
    Vector,
    Hybrid,
}

/// Indexing and search errors.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("{0}")]
    Message(String),
}

/// Main index handle.
pub struct Index {
    storage: IndexStorage,
    manifest: Manifest,
    segments: Vec<SegmentView>,
    config: IndexConfig,
    embedding_provider: Box<dyn EmbeddingProvider>,
}

impl Index {
    /// Opens an index from disk.
    ///
    /// # Arguments
    ///
    /// * `path` - Index root path.
    /// * `embedding_provider` - Embedding backend used for vector search.
    ///
    /// # Returns
    ///
    /// A loaded index instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest or any segment cannot be read.
    pub fn open(
        path: impl Into<PathBuf>,
        embedding_provider: Box<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        let storage = IndexStorage::open(path)?;
        let mut manifest = storage.load_manifest()?;
        storage.recover(&mut manifest)?;
        let mut segments = Vec::new();
        for seg in &manifest.segments {
            segments.push(storage.read_segment(seg)?);
        }
        Ok(Self {
            storage,
            manifest,
            segments,
            config: IndexConfig::default(),
            embedding_provider,
        })
    }

    /// Returns the current manifest snapshot.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Indexes a batch of documents as a new immutable segment.
    pub fn index_documents(&mut self, docs: Vec<OkfDocument>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let logical_keys: Vec<String> = docs.iter().map(|d| d.logical_key.clone()).collect();
        self.delete_logical_keys(&logical_keys)?;

        let incoming_doc_ids: HashSet<String> = docs.iter().map(|d| d.doc_id.clone()).collect();
        self.manifest
            .tombstones
            .retain(|id| !incoming_doc_ids.contains(id));

        let texts: Vec<String> = docs.iter().map(|doc| doc.searchable_text.clone()).collect();
        let embeddings = self.embedding_provider.embed(&texts)?;
        let pairs: Vec<(String, String)> = docs
            .iter()
            .map(|d| (d.doc_id.clone(), d.searchable_text.clone()))
            .collect();
        let bm25 = Bm25Index::build(&pairs, self.config.bm25.clone());
        let segment_id = format!("seg_{:016x}", now_nanos());
        let metadata = SegmentMetadata {
            doc_count: docs.len(),
            avg_doc_len: bm25.avg_doc_len,
            embedding_dimension: self.embedding_provider.dimension(),
            created_at: now_nanos() as u64,
        };
        let hnsw = if self.config.ann.enabled && docs.len() >= self.config.ann.threshold {
            Some(crate::ann::HnswIndex::build(&embeddings, &self.config.ann))
        } else {
            None
        };
        let segment = SegmentFile {
            segment_id: segment_id.clone(),
            metadata,
            documents: docs,
            bm25,
            embeddings,
            hnsw,
        };
        let entry = self.storage.write_segment(&segment)?;
        self.manifest.generation += 1;
        self.manifest.embedding_dimension = self.embedding_provider.dimension();
        self.manifest.embedding_model = "fastembed".to_string();
        self.manifest.bm25 = self.config.bm25.clone();
        self.manifest.segments.push(entry.clone());
        self.storage.save_manifest(&self.manifest)?;
        self.storage.clear_journal()?;
        self.segments.push(self.storage.read_segment(&entry)?);
        Ok(())
    }

    /// Tombstones document IDs.
    pub fn delete_doc_ids(&mut self, doc_ids: &[String]) -> Result<()> {
        for doc_id in doc_ids {
            if !self
                .manifest
                .tombstones
                .iter()
                .any(|existing| existing == doc_id)
            {
                self.manifest.tombstones.push(doc_id.clone());
            }
        }
        self.manifest.generation += 1;
        self.storage.save_manifest(&self.manifest)?;
        Ok(())
    }

    /// Tombstones documents by logical key.
    pub fn delete_logical_keys(&mut self, logical_keys: &[String]) -> Result<()> {
        let mut ids_to_delete = Vec::new();
        for segment in &self.segments {
            for doc in segment.document_views() {
                if logical_keys
                    .iter()
                    .any(|key| key == doc.logical_key().unwrap_or_default())
                {
                    ids_to_delete.push(doc.doc_id().unwrap_or_default().to_string());
                }
            }
        }
        self.delete_doc_ids(&ids_to_delete)
    }

    /// Replaces existing documents with new versions.
    pub fn update_documents(&mut self, docs: Vec<OkfDocument>) -> Result<()> {
        self.index_documents(docs)
    }

    /// Searches the index.
    pub fn search(
        &self,
        query: &str,
        mode: SearchMode,
        top_k: usize,
    ) -> Result<(Vec<SearchResult>, QueryPlan)> {
        let (results, plan, _) = self.search_advanced(query, mode, top_k, 0, None)?;
        Ok((results, plan))
    }

    /// Searches the index with pagination (offset & top_k) and optional metadata filtering.
    pub fn search_advanced(
        &self,
        query: &str,
        mode: SearchMode,
        top_k: usize,
        offset: usize,
        filter: Option<&QueryFilter>,
    ) -> Result<(Vec<SearchResult>, QueryPlan, usize)> {
        let query_embedding = if matches!(mode, SearchMode::Vector | SearchMode::Hybrid) {
            self.embedding_provider
                .embed(&[query.to_string()])?
                .into_iter()
                .next()
        } else {
            None
        };

        let mut valid_doc_ids = HashSet::new();
        let mut lexical_map: HashMap<String, f32> = HashMap::new();
        let mut vector_map: HashMap<String, f32> = HashMap::new();
        let mut all_docs = Vec::new();

        for segment in &self.segments {
            let segment_docs = segment.document_views();
            let eligible: Vec<bool> = segment_docs
                .iter()
                .map(|doc| {
                    let doc_id = doc.doc_id().unwrap_or_default();
                    !self.manifest.tombstones.iter().any(|dead| dead == doc_id)
                        && filter.is_none_or(|criteria| matches_filter(doc, criteria))
                })
                .collect();

            for (doc, is_eligible) in segment_docs.iter().zip(&eligible) {
                let doc_id = doc.doc_id().unwrap_or_default();
                if !is_eligible {
                    continue;
                }
                valid_doc_ids.insert(doc_id.to_string());
                all_docs.push(doc.to_owned());
            }

            if matches!(mode, SearchMode::Lexical | SearchMode::Hybrid) {
                for (doc_id, score) in segment.bm25().score(query) {
                    if valid_doc_ids.contains(&doc_id) {
                        *lexical_map.entry(doc_id).or_default() += score;
                    }
                }
            }
            if let Some(query_embedding) = &query_embedding {
                let candidate_limit = top_k.saturating_add(offset).max(1);
                let scores = cosine_scores(
                    query_embedding,
                    segment,
                    &self.config.ann,
                    candidate_limit,
                    &eligible,
                );
                for (doc_id, score) in scores {
                    if valid_doc_ids.contains(&doc_id) {
                        *vector_map.entry(doc_id).or_default() += score;
                    }
                }
            }
        }

        let lexical_order = sort_scores(&lexical_map);
        let vector_order = sort_scores(&vector_map);
        let fused = match mode {
            SearchMode::Lexical => lexical_order.clone(),
            SearchMode::Vector => vector_order.clone(),
            SearchMode::Hybrid => rrf_fuse(&lexical_order, &vector_order, self.config.rrf_k),
        };
        let results = results_from_docs(&all_docs, &lexical_map, &vector_map, &fused, query);
        let total_hits = results.len();
        let paginated = results.into_iter().skip(offset).take(top_k).collect();

        Ok((
            paginated,
            QueryPlan {
                query: query.to_string(),
                lexical_candidates: lexical_order,
                vector_candidates: vector_order,
                fused,
            },
            total_hits,
        ))
    }

    /// Indexes all documents in a bundle directory.
    pub fn add_bundle_dir(&mut self, bundle_dir: impl AsRef<Path>) -> Result<()> {
        let docs = crate::okf::load_bundle(bundle_dir.as_ref())?;
        self.update_documents(docs)
    }

    /// Compacts the current live segments into a fresh segment.
    pub fn compact(&mut self) -> Result<()> {
        let entry = self.storage.compact(&self.segments, &self.manifest)?;
        self.manifest.generation += 1;
        self.manifest.segments = vec![entry.clone()];
        self.manifest.tombstones.clear();
        self.storage.save_manifest(&self.manifest)?;
        let rebuilt = self.storage.read_segment(&entry)?;
        self.segments = vec![rebuilt];
        Ok(())
    }
}

fn matches_filter(
    doc: &crate::storage::DocumentView<'_>,
    filter: &crate::query::QueryFilter,
) -> bool {
    if !filter.types.is_empty() {
        let doc_type = doc.type_name().unwrap_or_default();
        if !filter
            .types
            .iter()
            .any(|t| t.eq_ignore_ascii_case(doc_type))
        {
            return false;
        }
    }
    if !filter.tags.is_empty() {
        let doc_tags = doc.tags();
        for tag in &filter.tags {
            if !doc_tags.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
                return false;
            }
        }
    }
    if let Some(prefix) = &filter.concept_path_prefix {
        let concept_path = doc.concept_path().unwrap_or_default();
        if !concept_path.starts_with(prefix) {
            return false;
        }
    }
    true
}

fn cosine_scores(
    query: &[f32],
    segment: &SegmentView,
    config: &crate::ann::AnnConfig,
    candidate_limit: usize,
    eligible: &[bool],
) -> Vec<(String, f32)> {
    let docs = segment.document_views();

    if config.enabled && docs.len() >= config.threshold {
        if let Some(hnsw) = segment.hnsw() {
            let embeddings = segment.embeddings();
            let ann_results = hnsw.search(query, candidate_limit, &embeddings);
            let filtered = ann_results
                .into_iter()
                .filter_map(|(idx, score)| {
                    if !eligible.get(idx).copied().unwrap_or(false) {
                        return None;
                    }
                    let doc_id = docs.get(idx)?.doc_id()?.to_string();
                    Some((doc_id, score))
                })
                .collect::<Vec<_>>();

            // A filtered ANN beam can contain too few eligible documents. Fall
            // back to exact scoring so filters and tombstones never reduce the
            // result set merely because excluded vectors occupied the beam.
            if filtered.len() >= candidate_limit {
                return filtered;
            }
        }
    }

    let vectors = segment.embeddings();
    let mut scores = Vec::with_capacity(vectors.len());
    for (idx, vec) in vectors.iter().enumerate() {
        if !eligible.get(idx).copied().unwrap_or(false) {
            continue;
        }
        scores.push((
            docs[idx].doc_id().unwrap_or_default().to_string(),
            cosine_similarity(query, vec),
        ));
    }
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(candidate_limit);
    scores
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-6)
}

fn sort_scores(map: &HashMap<String, f32>) -> Vec<(String, f32)> {
    let mut v: Vec<_> = map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// Opens an index handle.
pub fn open_index(
    path: impl Into<PathBuf>,
    embedding_provider: Box<dyn EmbeddingProvider>,
) -> Result<Index> {
    Index::open(path, embedding_provider)
}
