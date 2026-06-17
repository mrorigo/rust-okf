use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OkfDocument {
    pub logical_key: String,
    pub bundle_path: String,
    pub concept_path: String,
    pub file_path: String,
    pub doc_id: String,
    pub type_name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub resource: Option<String>,
    pub tags: Vec<String>,
    pub timestamp: Option<String>,
    pub body: String,
    pub searchable_text: String,
}

#[derive(Default)]
pub struct OkfDocumentBuilder {
    bundle_path: PathBuf,
    file_path: PathBuf,
    frontmatter: HashMap<String, serde_json::Value>,
    body: String,
}

impl OkfDocumentBuilder {
    pub fn new(bundle_path: impl Into<PathBuf>, file_path: impl Into<PathBuf>) -> Self {
        Self {
            bundle_path: bundle_path.into(),
            file_path: file_path.into(),
            ..Default::default()
        }
    }

    pub fn frontmatter_value(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.frontmatter.insert(key.into(), value.into());
        self
    }

    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    pub fn build(self) -> OkfDocument {
        let concept_path = self
            .file_path
            .strip_prefix(&self.bundle_path)
            .unwrap_or(&self.file_path)
            .to_string_lossy()
            .trim_start_matches(std::path::MAIN_SEPARATOR)
            .trim_end_matches(".md")
            .replace(std::path::MAIN_SEPARATOR, "/");

        let title = self.frontmatter.get("title").and_then(|v| v.as_str()).map(str::to_owned);
        let description = self.frontmatter.get("description").and_then(|v| v.as_str()).map(str::to_owned);
        let resource = self.frontmatter.get("resource").and_then(|v| v.as_str()).map(str::to_owned);
        let tags = self
            .frontmatter
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let type_name = self
            .frontmatter
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_owned();
        let timestamp = self.frontmatter.get("timestamp").and_then(|v| v.as_str()).map(str::to_owned);
        let searchable_text = build_searchable_text(title.as_deref(), description.as_deref(), &tags, &self.body);
        let doc_id = doc_id_for(&self.bundle_path, &self.file_path, &self.body);
        let logical_key = logical_key_for(&self.bundle_path, &concept_path);

        OkfDocument {
            logical_key,
            bundle_path: self.bundle_path.to_string_lossy().to_string(),
            concept_path,
            file_path: self.file_path.to_string_lossy().to_string(),
            doc_id,
            type_name,
            title,
            description,
            resource,
            tags,
            timestamp,
            body: self.body,
            searchable_text,
        }
    }
}

pub fn build_searchable_text(title: Option<&str>, description: Option<&str>, tags: &[String], body: &str) -> String {
    let mut parts = Vec::new();
    if let Some(title) = title {
        parts.push(title.to_string());
    }
    if let Some(description) = description {
        parts.push(description.to_string());
    }
    if !tags.is_empty() {
        parts.push(tags.join(" "));
    }
    parts.push(body.to_string());
    parts.join("\n")
}

pub fn doc_id_for(bundle_path: &Path, file_path: &Path, body: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bundle_path.to_string_lossy().as_bytes());
    hasher.update(b"\0");
    hasher.update(file_path.to_string_lossy().as_bytes());
    hasher.update(b"\0");
    hasher.update(body.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn logical_key_for(bundle_path: &Path, concept_path: &str) -> String {
    format!(
        "{}::{}",
        bundle_path.to_string_lossy(),
        concept_path.replace('\\', "/")
    )
}

pub fn load_bundle(bundle_dir: &Path) -> anyhow::Result<Vec<OkfDocument>> {
    let mut docs = Vec::new();
    for entry in walkdir::WalkDir::new(bundle_dir).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let content = fs::read_to_string(entry.path())?;
        let (frontmatter, body) = split_frontmatter(&content)?;
        let mut builder = OkfDocumentBuilder::new(bundle_dir, entry.path()).body(body);
        for (k, v) in frontmatter {
            builder = builder.frontmatter_value(k, v);
        }
        docs.push(builder.build());
    }
    Ok(docs)
}

fn split_frontmatter(content: &str) -> anyhow::Result<(HashMap<String, serde_json::Value>, String)> {
    if !content.starts_with("---\n") {
        return Ok((HashMap::new(), content.to_string()));
    }
    let mut lines = content.lines();
    let _ = lines.next();
    let mut yaml_lines = Vec::new();
    for line in &mut lines {
        if line.trim() == "---" {
            break;
        }
        yaml_lines.push(line);
    }
    let body = lines.collect::<Vec<_>>().join("\n");
    let yaml_text = yaml_lines.join("\n");
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml_text)?;
    let mut map = HashMap::new();
    if let serde_yaml::Value::Mapping(mapping) = value {
        for (k, v) in mapping {
            if let Some(k) = k.as_str() {
                map.insert(k.to_string(), serde_yaml_to_json(v));
            }
        }
    }
    Ok((map, body))
}

fn serde_yaml_to_json(value: serde_yaml::Value) -> serde_json::Value {
    match value {
        serde_yaml::Value::Bool(b) => serde_json::Value::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                serde_json::Value::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                serde_json::Value::Number(u.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f).map_or(serde_json::Value::Null, serde_json::Value::Number)
            } else {
                serde_json::Value::Null
            }
        }
        serde_yaml::Value::String(s) => serde_json::Value::String(s),
        serde_yaml::Value::Sequence(seq) => serde_json::Value::Array(seq.into_iter().map(serde_yaml_to_json).collect()),
        _ => serde_json::Value::Null,
    }
}
