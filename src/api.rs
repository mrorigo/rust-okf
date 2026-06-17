use crate::embedding::FastEmbedProvider;
use crate::index::{Index, SearchMode};
use crate::okf::OkfDocumentBuilder;
use crate::schema::{DeleteRequest, DocumentInput, SearchModeRequest, SearchRequest, SearchResponse, StatusResponse};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use utoipa::OpenApi;

#[derive(Clone)]
pub struct ApiState {
    pub index: Arc<Mutex<Index>>,
}

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/openapi.json", get(openapi_json))
        .route("/search", post(search))
        .route("/documents", post(index_documents))
        .route("/documents/update", post(update_documents))
        .route("/documents/delete", post(delete_documents))
        .with_state(state)
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Service is ready", body = StatusResponse)
    )
)]
pub async fn health() -> (StatusCode, Json<StatusResponse>) {
    (StatusCode::OK, Json(StatusResponse { status: "ok".to_string() }))
}

#[utoipa::path(
    get,
    path = "/openapi.json",
    responses(
        (status = 200, description = "OpenAPI schema")
    )
)]
pub async fn openapi_json() -> (StatusCode, Json<serde_json::Value>) {
    let doc = crate::openapi::ApiDoc::openapi();
    (
        StatusCode::OK,
        Json(serde_json::to_value(doc).unwrap_or_else(|_| serde_json::json!({}))),
    )
}

#[utoipa::path(
    post,
    path = "/search",
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = SearchResponse)
    )
)]
pub async fn search(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<StatusResponse>)> {
    let mode = match req.mode.unwrap_or(SearchModeRequest::Hybrid) {
        SearchModeRequest::Lexical => SearchMode::Lexical,
        SearchModeRequest::Vector => SearchMode::Vector,
        SearchModeRequest::Hybrid => SearchMode::Hybrid,
    };
    let index = state
        .index
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: "index lock poisoned".to_string() })))?;
    let (results, plan) = index
        .search(&req.query, mode, req.top_k.unwrap_or(10))
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: err.to_string() })))?;
    Ok(Json(SearchResponse { results, plan }))
}

fn build_document(doc: DocumentInput) -> crate::okf::OkfDocument {
    let mut builder = OkfDocumentBuilder::new(PathBuf::from(&doc.bundle_path), PathBuf::from(&doc.file_path)).body(doc.body);
    for (k, v) in doc.frontmatter {
        builder = builder.frontmatter_value(k, v);
    }
    builder.build()
}

#[utoipa::path(
    post,
    path = "/documents",
    request_body = Vec<DocumentInput>,
    responses(
        (status = 200, description = "Documents indexed", body = StatusResponse)
    )
)]
pub async fn index_documents(
    State(state): State<ApiState>,
    Json(docs): Json<Vec<DocumentInput>>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<StatusResponse>)> {
    let built = docs.into_iter().map(build_document).collect::<Vec<_>>();
    let mut index = state
        .index
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: "index lock poisoned".to_string() })))?;
    index
        .index_documents(built)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: err.to_string() })))?;
    Ok(Json(StatusResponse { status: "ok".to_string() }))
}

#[utoipa::path(
    post,
    path = "/documents/update",
    request_body = Vec<DocumentInput>,
    responses(
        (status = 200, description = "Documents updated", body = StatusResponse)
    )
)]
pub async fn update_documents(
    State(state): State<ApiState>,
    Json(docs): Json<Vec<DocumentInput>>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<StatusResponse>)> {
    let built = docs.into_iter().map(build_document).collect::<Vec<_>>();
    let mut index = state
        .index
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: "index lock poisoned".to_string() })))?;
    index
        .update_documents(built)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: err.to_string() })))?;
    Ok(Json(StatusResponse { status: "ok".to_string() }))
}

#[utoipa::path(
    post,
    path = "/documents/delete",
    request_body = DeleteRequest,
    responses(
        (status = 200, description = "Documents deleted", body = StatusResponse)
    )
)]
pub async fn delete_documents(
    State(state): State<ApiState>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<StatusResponse>)> {
    let mut index = state
        .index
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: "index lock poisoned".to_string() })))?;
    if !req.logical_keys.is_empty() {
        index
            .delete_logical_keys(&req.logical_keys)
            .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: err.to_string() })))?;
    }
    if !req.doc_ids.is_empty() {
        index
            .delete_doc_ids(&req.doc_ids)
            .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, Json(StatusResponse { status: err.to_string() })))?;
    }
    Ok(Json(StatusResponse { status: "ok".to_string() }))
}

pub async fn serve(index: Index, bind: String) -> anyhow::Result<()> {
    let state = ApiState {
        index: Arc::new(Mutex::new(index)),
    };
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn default_provider() -> anyhow::Result<FastEmbedProvider> {
    FastEmbedProvider::new_default()
}
