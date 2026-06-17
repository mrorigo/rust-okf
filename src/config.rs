/// Rust guideline compliant 2026-06-17
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// FastEmbed-related configuration.
///
/// # Notes
///
/// This controls whether the production embedding backend is enabled and which
/// Hugging Face model identifier should be used when FastEmbed is active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEmbedConfig {
    /// Enables the production embedding backend.
    pub enabled: bool,
    /// Hugging Face model identifier used by FastEmbed.
    pub model: String,
}

impl Default for FastEmbedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "BAAI/bge-small-en-v1.5".to_string(),
        }
    }
}

/// Application configuration for the CLI and HTTP server.
///
/// # Notes
///
/// The configuration is stored in TOML format and provides the default index
/// path, bind address, and embedding backend settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Embedding backend configuration.
    #[serde(default)]
    pub fastembed: FastEmbedConfig,
    /// Default HTTP bind address.
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Default index path.
    #[serde(default = "default_index")]
    pub index: String,
}

fn default_bind() -> String {
    "127.0.0.1:8787".to_string()
}

fn default_index() -> String {
    "./okf-index".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            fastembed: FastEmbedConfig::default(),
            bind: default_bind(),
            index: default_index(),
        }
    }
}

impl AppConfig {
    /// Loads configuration from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to read.
    ///
    /// # Returns
    ///
    /// The parsed configuration, or the default configuration if the file does
    /// not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    /// Writes the default configuration to disk.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_default(path: impl AsRef<Path>) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(&Self::default())?;
        fs::write(path, text)?;
        Ok(())
    }
}
