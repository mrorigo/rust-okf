use crate::bm25::{Bm25Config, Bm25Index};
use crate::embedding::EmbeddingProvider;
use crate::okf::OkfDocument;
use crate::query::{rrf_fuse, QueryPlan, SearchResult};
use crate::storage::{results_from_docs, IndexStorage, Manifest, SegmentFile, SegmentMetadata};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub bm25: Bm25Config,
    pub rrf_k: f32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            bm25: Bm25Config::default(),
            rrf_k: 60.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Lexical,
    Vector,
    Hybrid,
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("{0}")]
    Message(String),
}

pub struct Index {
    storage: IndexStorage,
    manifest: Manifest,
    segments: Vec<SegmentFile>,
    config: IndexConfig,
    embedding_provider: Box<dyn EmbeddingProvider>,
}

impl Index {
    pub fn open(path: impl Into<PathBuf>, embedding_provider: Box<dyn EmbeddingProvider>) -> Result<Self> {
        let storage = IndexStorage::open(path)?;
        let manifest = storage.load_manifest()?;
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

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn index_documents(&mut self, docs: Vec<OkfDocument>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let tombstoned_keys: HashSet<String> = self.manifest.tombstones.iter().cloned().collect();
        let docs: Vec<OkfDocument> = docs
            .into_iter()
            .filter(|doc| !tombstoned_keys.contains(&doc.doc_id))
            .collect();
        if docs.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = docs.iter().map(|doc| doc.searchable_text.clone()).collect();
        let embeddings = self.embedding_provider.embed(&texts)?;
        let pairs: Vec<(String, String)> = docs.iter().map(|d| (d.doc_id.clone(), d.searchable_text.clone())).collect();
        let bm25 = Bm25Index::build(&pairs, self.config.bm25.clone());
        let segment_id = format!("seg_{:016x}", now_nanos());
        let metadata = SegmentMetadata {
            doc_count: docs.len(),
            avg_doc_len: bm25.avg_doc_len,
            embedding_dimension: self.embedding_provider.dimension(),
            created_at: now_nanos() as u64,
        };
        let segment = SegmentFile {
            segment_id: segment_id.clone(),
            metadata,
            documents: docs,
            bm25,
            embeddings,
        };
        let entry = self.storage.write_segment(&segment)?;
        self.manifest.generation += 1;
        self.manifest.embedding_dimension = self.embedding_provider.dimension();
        self.manifest.embedding_model = "fastembed".to_string();
        self.manifest.bm25 = self.config.bm25.clone();
        self.manifest.segments.push(entry);
        self.storage.save_manifest(&self.manifest)?;
        self.segments.push(segment);
        Ok(())
    }

    pub fn delete_doc_ids(&mut self, doc_ids: &[String]) -> Result<()> {
        for doc_id in doc_ids {
            if !self.manifest.tombstones.iter().any(|existing| existing == doc_id) {
                self.manifest.tombstones.push(doc_id.clone());
            }
        }
        self.manifest.generation += 1;
        self.storage.save_manifest(&self.manifest)?;
        Ok(())
    }

    pub fn delete_logical_keys(&mut self, logical_keys: &[String]) -> Result<()> {
        let mut ids_to_delete = Vec::new();
        for segment in &self.segments {
            for doc in &segment.documents {
                if logical_keys.iter().any(|key| key == &doc.logical_key) {
                    ids_to_delete.push(doc.doc_id.clone());
                }
            }
        }
        self.delete_doc_ids(&ids_to_delete)
    }

    pub fn update_documents(&mut self, docs: Vec<OkfDocument>) -> Result<()> {
        let logical_keys: Vec<String> = docs.iter().map(|d| d.logical_key.clone()).collect();
        self.delete_logical_keys(&logical_keys)?;
        self.index_documents(docs)
    }

    pub fn search(&self, query: &str, mode: SearchMode, top_k: usize) -> Result<(Vec<SearchResult>, QueryPlan)> {
        let query_embedding = if matches!(mode, SearchMode::Vector | SearchMode::Hybrid) {
            self.embedding_provider.embed(&[query.to_string()])?.into_iter().next()
        } else {
            None
        };

        let mut lexical_map: HashMap<String, f32> = HashMap::new();
        let mut vector_map: HashMap<String, f32> = HashMap::new();
        let mut all_docs = Vec::new();

        for segment in &self.segments {
            all_docs.extend(segment.documents.iter().cloned().filter(|doc| {
                !self.manifest.tombstones.iter().any(|dead| dead == &doc.doc_id)
            }));
            if matches!(mode, SearchMode::Lexical | SearchMode::Hybrid) {
                for (doc_id, score) in segment.bm25.score(query) {
                    if self.manifest.tombstones.iter().any(|dead| dead == &doc_id) {
                        continue;
                    }
                    *lexical_map.entry(doc_id).or_default() += score;
                }
            }
            if let Some(query_embedding) = &query_embedding {
                let scores = cosine_scores(query_embedding, &segment.embeddings, &segment.documents);
                for (doc_id, score) in scores {
                    if self.manifest.tombstones.iter().any(|dead| dead == &doc_id) {
                        continue;
                    }
                    *vector_map.entry(doc_id).or_default() += score;
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
        let mut results = results_from_docs(&all_docs, &lexical_map, &vector_map, &fused, query);
        results.truncate(top_k);
        Ok((
            results,
            QueryPlan {
                query: query.to_string(),
                lexical_candidates: lexical_order,
                vector_candidates: vector_order,
                fused,
            },
        ))
    }

    pub fn add_bundle_dir(&mut self, bundle_dir: impl AsRef<Path>) -> Result<()> {
        let docs = crate::okf::load_bundle(bundle_dir.as_ref())?;
        self.update_documents(docs)
    }
}

fn cosine_scores(query: &[f32], vectors: &[Vec<f32>], docs: &[OkfDocument]) -> Vec<(String, f32)> {
    let mut scores = Vec::with_capacity(vectors.len());
    for (idx, vec) in vectors.iter().enumerate() {
        scores.push((docs[idx].doc_id.clone(), cosine_similarity(query, vec)));
    }
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
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
}

pub fn open_index(path: impl Into<PathBuf>, embedding_provider: Box<dyn EmbeddingProvider>) -> Result<Index> {
    Index::open(path, embedding_provider)
}
