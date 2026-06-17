/// Rust guideline compliant 2026-06-17
use anyhow::Result;

/// Embeds a batch of text inputs into dense vectors.
///
/// # Notes
///
/// Implementations must be safe to share across threads.
pub trait EmbeddingProvider: Send + Sync {
    /// Embeds the provided texts.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Text inputs to embed.
    ///
    /// # Returns
    ///
    /// A vector per input, each with the provider's fixed dimensionality.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
    /// Returns the embedding dimensionality.
    fn dimension(&self) -> usize;
}

/// FastEmbed-backed production embedding provider.
pub struct FastEmbedProvider {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    dimension: usize,
}

impl FastEmbedProvider {
    /// Creates the default FastEmbed provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be initialized.
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
        let embeddings = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("fastembed model lock poisoned"))?
            .embed(texts, None)?;
        Ok(embeddings.into_iter().map(|v| v.to_vec()).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Deterministic embedding provider used for tests and offline workflows.
#[derive(Clone)]
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    /// Creates a new mock provider.
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
