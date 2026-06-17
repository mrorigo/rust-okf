use rust_okf::{open_index, MockEmbeddingProvider, OkfDocumentBuilder, SearchMode};
use std::fs;

#[test]
fn bm25_and_rrf_searches_documents() {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("bundle")).unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let doc1 = OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/a.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();
    let doc2 = OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/b.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Revenue")
        .body("Revenue from orders and invoices")
        .build();
    index
        .index_documents(vec![doc1.clone(), doc2.clone()])
        .unwrap();

    let (results, plan): (Vec<rust_okf::query::SearchResult>, _) =
        index.search("orders", SearchMode::Hybrid, 10).unwrap();
    assert!(!results.is_empty());
    assert_eq!(plan.query, "orders");
    assert!(!plan.lexical_candidates.is_empty());
}

#[test]
fn tombstones_hide_deleted_documents() {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("bundle")).unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let doc = OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/a.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();
    let doc_id = doc.doc_id.clone();
    index.index_documents(vec![doc]).unwrap();
    index.delete_doc_ids(&[doc_id]).unwrap();
    let (results, _) = index.search("orders", SearchMode::Hybrid, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn load_bundle_reads_markdown_with_frontmatter() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::write(
        bundle.join("a.md"),
        "---\ntype: Metric\ntitle: Orders\ntags:\n  - sales\n---\nhello world\n",
    )
    .unwrap();

    let docs = rust_okf::okf::load_bundle(&bundle).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].title.as_deref(), Some("Orders"));
    assert_eq!(docs[0].tags, vec!["sales"]);
}
