/// Rust guideline compliant 2026-06-17
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// BM25 parameter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Term frequency saturation parameter.
    pub k1: f32,
    /// Document length normalization parameter.
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// In-memory BM25 index for a segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    /// Scoring configuration.
    pub config: Bm25Config,
    /// Document lengths by document ID.
    pub doc_len: HashMap<String, usize>,
    /// Average document length across the corpus.
    pub avg_doc_len: f32,
    /// Document count in the corpus.
    pub doc_count: usize,
    /// Term document frequency map.
    pub term_doc_freq: HashMap<String, usize>,
    /// Posting lists keyed by term.
    pub postings: HashMap<String, HashMap<String, usize>>,
}

impl Bm25Index {
    /// Builds a BM25 index from `(doc_id, text)` pairs.
    pub fn build(docs: &[(String, String)], config: Bm25Config) -> Self {
        let mut doc_len = HashMap::new();
        let mut term_doc_freq = HashMap::new();
        let mut postings: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut total_len = 0usize;

        for (doc_id, text) in docs {
            let terms = tokenize(text);
            total_len += terms.len();
            doc_len.insert(doc_id.clone(), terms.len());
            let mut seen = HashSet::new();
            for term in terms {
                let entry = postings.entry(term.clone()).or_default();
                *entry.entry(doc_id.clone()).or_default() += 1;
                if seen.insert(term.clone()) {
                    *term_doc_freq.entry(term).or_default() += 1;
                }
            }
        }

        Self {
            config,
            avg_doc_len: if docs.is_empty() {
                0.0
            } else {
                total_len as f32 / docs.len() as f32
            },
            doc_count: docs.len(),
            doc_len,
            term_doc_freq,
            postings,
        }
    }

    /// Scores a query against the index.
    ///
    /// # Arguments
    ///
    /// * `query` - Query text.
    ///
    /// # Returns
    ///
    /// Document scores keyed by document ID.
    pub fn score(&self, query: &str) -> HashMap<String, f32> {
        let mut scores = HashMap::new();
        let q_terms = tokenize(query);
        for term in q_terms {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let df = *self.term_doc_freq.get(&term).unwrap_or(&0) as f32;
            if df == 0.0 || self.doc_count == 0 {
                continue;
            }
            let idf = ((self.doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (doc_id, tf) in posting {
                let dl = *self.doc_len.get(doc_id).unwrap_or(&0) as f32;
                let tf = *tf as f32;
                let denom = tf
                    + self.config.k1
                        * (1.0 - self.config.b + self.config.b * (dl / self.avg_doc_len.max(1e-6)));
                let score = idf * (tf * (self.config.k1 + 1.0)) / denom;
                *scores.entry(doc_id.clone()).or_default() += score;
            }
        }
        scores
    }
}

/// Tokenizes text into lowercase alphanumeric terms.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}
