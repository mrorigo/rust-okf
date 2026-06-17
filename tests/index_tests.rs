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

#[test]
fn reopen_index_reads_custom_segment_format() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();
    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    let doc = OkfDocumentBuilder::new(&bundle, bundle.join("a.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();
    index.index_documents(vec![doc]).unwrap();

    let reopened = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let (results, _) = reopened.search("orders", SearchMode::Hybrid, 10).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn recovery_clears_staging_segments() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(index_dir.join("segments").join("seg_dead.staging")).unwrap();
    fs::create_dir_all(index_dir.join("journal")).unwrap();
    fs::write(
        index_dir.join("journal").join("journal.bin"),
        bincode::serde::encode_to_vec(
            rust_okf::storage::Journal {
                format_version: rust_okf::storage::INDEX_FORMAT_VERSION,
                entries: vec![rust_okf::storage::JournalEntry::BeginCommit {
                    segment_id: "seg_dead".to_string(),
                }],
            },
            bincode::config::standard(),
        )
        .unwrap(),
    )
    .unwrap();

    let _ = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    assert!(!index_dir.join("segments").join("seg_dead.staging").exists());
}

#[test]
fn compaction_preserves_live_results() {
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
    let doc2_id = doc2.doc_id.clone();
    index.index_documents(vec![doc1, doc2]).unwrap();
    index.delete_doc_ids(&[doc2_id]).unwrap();
    let before = index.search("orders", SearchMode::Hybrid, 10).unwrap().0;
    index.compact().unwrap();
    let after = index.search("orders", SearchMode::Hybrid, 10).unwrap().0;
    assert_eq!(
        before.iter().map(|r| r.doc_id.clone()).collect::<Vec<_>>(),
        after.iter().map(|r| r.doc_id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn golden_query_order_is_stable() {
    let tmp = tempfile::tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("bundle")).unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let docs = vec![
        OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/a.md"))
            .frontmatter_value("type", "Metric")
            .frontmatter_value("title", "Orders")
            .body("Orders completed by customers")
            .build(),
        OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/b.md"))
            .frontmatter_value("type", "Metric")
            .frontmatter_value("title", "Revenue")
            .body("Revenue from orders and invoices")
            .build(),
        OkfDocumentBuilder::new(tmp.path().join("bundle"), tmp.path().join("bundle/c.md"))
            .frontmatter_value("type", "Metric")
            .frontmatter_value("title", "Customers")
            .body("Customers who ordered and paid")
            .build(),
    ];
    let expected = vec![
        "Orders".to_string(),
        "Revenue".to_string(),
        "Customers".to_string(),
    ];
    index.index_documents(docs).unwrap();
    let (results, _) = index.search("orders", SearchMode::Hybrid, 3).unwrap();
    let actual: Vec<String> = results
        .iter()
        .map(|r| r.title.clone().unwrap_or_default())
        .collect();
    assert_eq!(actual, expected);
}
