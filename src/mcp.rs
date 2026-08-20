use crate::{Index, QueryFilter, SearchMode};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{schemars, tool, tool_router, ErrorData as McpError, ServiceExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    pub query: String,
    pub mode: Option<String>,
    pub top_k: Option<usize>,
    pub offset: Option<usize>,
    pub filter: Option<QueryFilterParams>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct QueryFilterParams {
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub concept_path_prefix: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DocumentsParams {
    pub documents: Vec<McpDocumentInput>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct McpDocumentInput {
    pub bundle_path: String,
    pub file_path: String,
    #[serde(default)]
    pub frontmatter: serde_json::Map<String, serde_json::Value>,
    pub body: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteParams {
    #[serde(default)]
    pub doc_ids: Vec<String>,
    #[serde(default)]
    pub logical_keys: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SearchOutput {
    results: Vec<crate::query::SearchResult>,
    total_hits: usize,
    offset: usize,
    top_k: usize,
}

#[derive(Debug, Serialize)]
struct MutationOutput {
    status: &'static str,
    count: usize,
}

#[derive(Clone)]
pub struct McpServer {
    index: Arc<Mutex<Index>>,
}

impl McpServer {
    pub fn new(index: Index) -> Self {
        Self {
            index: Arc::new(Mutex::new(index)),
        }
    }
}

fn tool_error(error: impl std::fmt::Display) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

fn parse_mode(mode: Option<&str>) -> Result<SearchMode, McpError> {
    match mode.unwrap_or("hybrid") {
        "lexical" => Ok(SearchMode::Lexical),
        "vector" => Ok(SearchMode::Vector),
        "hybrid" => Ok(SearchMode::Hybrid),
        value => Err(McpError::invalid_params(
            format!("unsupported search mode: {value}"),
            None,
        )),
    }
}

fn make_filter(filter: Option<QueryFilterParams>) -> Option<QueryFilter> {
    filter.map(|value| QueryFilter {
        types: value.types,
        tags: value.tags,
        concept_path_prefix: value.concept_path_prefix,
    })
}

fn json_result<T: Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string(value).map_err(tool_error)?;
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

#[tool_router(server_handler)]
impl McpServer {
    #[tool(description = "Search indexed OKF documents")]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.query.trim().is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        let mode = parse_mode(params.mode.as_deref())?;
        let top_k = params.top_k.unwrap_or(10);
        let offset = params.offset.unwrap_or(0);
        let filter = make_filter(params.filter);
        let index = self.index.lock().map_err(tool_error)?;
        let (results, _plan, total_hits) = index
            .search_advanced(&params.query, mode, top_k, offset, filter.as_ref())
            .map_err(tool_error)?;
        json_result(&SearchOutput {
            results,
            total_hits,
            offset,
            top_k,
        })
    }

    #[tool(description = "Index documents into the OKF index")]
    async fn index_documents(
        &self,
        Parameters(params): Parameters<DocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let count = params.documents.len();
        let documents = params
            .documents
            .into_iter()
            .map(build_document)
            .collect::<Vec<_>>();
        self.index
            .lock()
            .map_err(tool_error)?
            .index_documents(documents)
            .map_err(tool_error)?;
        json_result(&MutationOutput {
            status: "ok",
            count,
        })
    }

    #[tool(description = "Update documents in the OKF index")]
    async fn update_documents(
        &self,
        Parameters(params): Parameters<DocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let count = params.documents.len();
        let documents = params
            .documents
            .into_iter()
            .map(build_document)
            .collect::<Vec<_>>();
        self.index
            .lock()
            .map_err(tool_error)?
            .update_documents(documents)
            .map_err(tool_error)?;
        json_result(&MutationOutput {
            status: "ok",
            count,
        })
    }

    #[tool(description = "Delete documents by document ID or logical key")]
    async fn delete_documents(
        &self,
        Parameters(params): Parameters<DeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.doc_ids.is_empty() && params.logical_keys.is_empty() {
            return Err(McpError::invalid_params(
                "doc_ids or logical_keys must not be empty",
                None,
            ));
        }
        let count = params.doc_ids.len() + params.logical_keys.len();
        let mut index = self.index.lock().map_err(tool_error)?;
        index
            .delete_doc_ids(&params.doc_ids)
            .and_then(|_| index.delete_logical_keys(&params.logical_keys))
            .map_err(tool_error)?;
        json_result(&MutationOutput {
            status: "ok",
            count,
        })
    }

    #[tool(description = "Compact the OKF index")]
    async fn compact(&self) -> Result<CallToolResult, McpError> {
        self.index
            .lock()
            .map_err(tool_error)?
            .compact()
            .map_err(tool_error)?;
        json_result(&MutationOutput {
            status: "ok",
            count: 0,
        })
    }

    #[tool(description = "Return OKF index status")]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        let index = self.index.lock().map_err(tool_error)?;
        let manifest = index.manifest();
        json_result(&serde_json::json!({
            "status": "ok",
            "generation": manifest.generation,
            "segments": manifest.segments.len(),
            "tombstones": manifest.tombstones.len(),
        }))
    }
}

fn build_document(doc: McpDocumentInput) -> crate::OkfDocument {
    let mut builder = crate::OkfDocumentBuilder::new(
        std::path::PathBuf::from(&doc.bundle_path),
        std::path::PathBuf::from(&doc.file_path),
    )
    .body(doc.body);
    for (key, value) in doc.frontmatter {
        builder = builder.frontmatter_value(key, value);
    }
    builder.build()
}

pub async fn serve(index: Index) -> anyhow::Result<()> {
    let service = McpServer::new(index)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_mode, SearchParams};
    use crate::SearchMode;

    #[test]
    fn defaults_to_hybrid_search() {
        assert!(matches!(parse_mode(None), Ok(SearchMode::Hybrid)));
    }

    #[test]
    fn rejects_unknown_search_mode() {
        assert!(parse_mode(Some("unknown")).is_err());
    }

    #[test]
    fn search_parameters_deserialize() {
        let params: SearchParams = serde_json::from_str(
            r#"{
            "query": "orders",
            "mode": "lexical",
            "top_k": 5,
            "offset": 2,
            "filter": {"types": ["Metric"]}
        }"#,
        )
        .unwrap();
        assert_eq!(params.query, "orders");
        assert_eq!(params.top_k, Some(5));
        assert_eq!(params.offset, Some(2));
    }
}
