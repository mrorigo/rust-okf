use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    pub config: Bm25Config,
    pub doc_len: HashMap<String, usize>,
    pub avg_doc_len: f32,
    pub doc_count: usize,
    pub term_doc_freq: HashMap<String, usize>,
    pub postings: HashMap<String, HashMap<String, usize>>,
}

impl Bm25Index {
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
            avg_doc_len: if docs.is_empty() { 0.0 } else { total_len as f32 / docs.len() as f32 },
            doc_count: docs.len(),
            doc_len,
            term_doc_freq,
            postings,
        }
    }

    pub fn score(&self, query: &str) -> HashMap<String, f32> {
        let mut scores = HashMap::new();
        let q_terms = tokenize(query);
        for term in q_terms {
            let Some(posting) = self.postings.get(&term) else { continue };
            let df = *self.term_doc_freq.get(&term).unwrap_or(&0) as f32;
            if df == 0.0 || self.doc_count == 0 {
                continue;
            }
            let idf = ((self.doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (doc_id, tf) in posting {
                let dl = *self.doc_len.get(doc_id).unwrap_or(&0) as f32;
                let tf = *tf as f32;
                let denom = tf + self.config.k1 * (1.0 - self.config.b + self.config.b * (dl / self.avg_doc_len.max(1e-6)));
                let score = idf * (tf * (self.config.k1 + 1.0)) / denom;
                *scores.entry(doc_id.clone()).or_default() += score;
            }
        }
        scores
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}
