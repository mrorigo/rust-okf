use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastEmbedConfig {
    pub enabled: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub fastembed: FastEmbedConfig,
    #[serde(default = "default_bind")]
    pub bind: String,
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
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    pub fn save_default(path: impl AsRef<Path>) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(&Self::default())?;
        fs::write(path, text)?;
        Ok(())
    }
}
