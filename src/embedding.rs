use anyhow::Result;

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

pub struct FastEmbedProvider {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    dimension: usize,
}

impl FastEmbedProvider {
    pub fn new_default() -> Result<Self> {
        use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
        let model = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::BGESmallENV15))?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
            dimension: 384,
        })
    }
}

impl EmbeddingProvider for FastEmbedProvider {
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let embeddings = self.model.lock().unwrap().embed(texts, None)?;
        Ok(embeddings.into_iter().map(|v| v.to_vec()).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[derive(Clone)]
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|input| {
                let mut v = vec![0.0; self.dimension];
                for (i, byte) in input.as_bytes().iter().enumerate() {
                    v[i % self.dimension] += (*byte as f32) / 255.0;
                }
                v
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
