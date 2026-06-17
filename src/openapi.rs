/// Rust guideline compliant 2026-06-17
use crate::schema::{
    DeleteRequest, DocumentInput, SearchModeRequest, SearchRequest, SearchResponse, StatusResponse,
};
use utoipa::OpenApi;

/// OpenAPI document for the HTTP API.
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::health,
        crate::api::search,
        crate::api::index_documents,
        crate::api::update_documents,
        crate::api::delete_documents,
    ),
    components(
        schemas(
            SearchRequest,
            SearchResponse,
            SearchModeRequest,
            DocumentInput,
            DeleteRequest,
            StatusResponse
        )
    ),
    tags(
        (name = "okf", description = "OKF indexing and hybrid search API")
    )
)]
pub struct ApiDoc;
