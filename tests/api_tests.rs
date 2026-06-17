use axum::body::Body;
use axum::http::{Request, StatusCode};
use rust_okf::api::{router, ApiState};
use rust_okf::embedding::MockEmbeddingProvider;
use rust_okf::index::open_index;
use rust_okf::okf::OkfDocumentBuilder;
use std::fs;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

fn build_index() -> rust_okf::Index {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();
    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    let doc = OkfDocumentBuilder::new(&bundle, bundle.join("orders.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();
    index.index_documents(vec![doc]).unwrap();
    index
}

#[tokio::test]
async fn health_and_search_routes_work() {
    let app = router(ApiState {
        index: Arc::new(Mutex::new(build_index())),
    });

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"query":"orders","mode":"hybrid","top_k":5}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
}

#[tokio::test]
async fn openapi_route_exposes_schema() {
    let app = router(ApiState {
        index: Arc::new(Mutex::new(build_index())),
    });
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
