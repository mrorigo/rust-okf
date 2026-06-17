use crate::schema::{DeleteRequest, DocumentInput, SearchRequest, SearchResponse, SearchModeRequest, StatusResponse};
use utoipa::OpenApi;

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
