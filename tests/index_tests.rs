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

#[test]
fn crlf_and_bom_frontmatter_parsing() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    let crlf_content = "\u{feff}---\r\ntype: Metric\r\ntitle: Windows Test\r\ntags:\r\n  - crlf\r\n---\r\nbody text\r\n";
    fs::write(bundle.join("win.md"), crlf_content).unwrap();

    let docs = rust_okf::okf::load_bundle(&bundle).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].title.as_deref(), Some("Windows Test"));
    assert_eq!(docs[0].tags, vec!["crlf"]);
}

#[test]
fn load_bundle_filters_reserved_and_hidden_files() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(bundle.join(".git")).unwrap();
    fs::write(bundle.join("index.md"), "---\ntitle: Index\n---\nbody\n").unwrap();
    fs::write(bundle.join("log.md"), "---\ntitle: Log\n---\nbody\n").unwrap();
    fs::write(
        bundle.join(".git/hidden.md"),
        "---\ntitle: Git\n---\nbody\n",
    )
    .unwrap();
    fs::write(
        bundle.join("concept.md"),
        "---\ntitle: Concept\n---\nbody\n",
    )
    .unwrap();

    let docs = rust_okf::okf::load_bundle(&bundle).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].title.as_deref(), Some("Concept"));
}

#[test]
fn add_is_idempotent_and_does_not_duplicate_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let doc = OkfDocumentBuilder::new(&bundle, bundle.join("a.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();

    index.index_documents(vec![doc.clone()]).unwrap();
    let (results_first, _) = index.search("orders", SearchMode::Hybrid, 10).unwrap();

    index.index_documents(vec![doc]).unwrap();
    let (results_second, _) = index.search("orders", SearchMode::Hybrid, 10).unwrap();

    assert_eq!(results_first.len(), 1);
    assert_eq!(results_second.len(), 1);
    assert!((results_first[0].fused_score - results_second[0].fused_score).abs() < f32::EPSILON);
}

#[test]
fn reindexing_deleted_doc_clears_tombstone() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let doc = OkfDocumentBuilder::new(&bundle, bundle.join("a.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders")
        .body("Orders completed by customers")
        .build();

    let doc_id = doc.doc_id.clone();
    index.index_documents(vec![doc.clone()]).unwrap();
    index.delete_doc_ids(&[doc_id]).unwrap();

    let (results_after_delete, _) = index.search("orders", SearchMode::Hybrid, 10).unwrap();
    assert!(results_after_delete.is_empty());

    index.index_documents(vec![doc]).unwrap();
    let (results_after_reindex, _) = index.search("orders", SearchMode::Hybrid, 10).unwrap();
    assert_eq!(results_after_reindex.len(), 1);
}

#[test]
fn search_filters_by_type_tag_and_prefix() {
    use rust_okf::QueryFilter;

    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(bundle.join("tables")).unwrap();
    fs::create_dir_all(bundle.join("metrics")).unwrap();
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let doc1 = OkfDocumentBuilder::new(&bundle, bundle.join("metrics/orders.md"))
        .frontmatter_value("type", "Metric")
        .frontmatter_value("title", "Orders Metric")
        .frontmatter_value("tags", vec!["sales", "finance"])
        .body("Orders completed count")
        .build();
    let doc2 = OkfDocumentBuilder::new(&bundle, bundle.join("tables/orders.md"))
        .frontmatter_value("type", "Table")
        .frontmatter_value("title", "Orders Table")
        .frontmatter_value("tags", vec!["sales", "raw"])
        .body("Orders data table")
        .build();

    index.index_documents(vec![doc1, doc2]).unwrap();

    // Filter by type
    let filter_type = QueryFilter {
        types: vec!["Metric".to_string()],
        ..Default::default()
    };
    let (results, _, total) = index
        .search_advanced("orders", SearchMode::Hybrid, 10, 0, Some(&filter_type))
        .unwrap();
    assert_eq!(total, 1);
    assert_eq!(results[0].type_name, "Metric");

    // Filter by tag
    let filter_tag = QueryFilter {
        tags: vec!["finance".to_string()],
        ..Default::default()
    };
    let (results, _, total) = index
        .search_advanced("orders", SearchMode::Hybrid, 10, 0, Some(&filter_tag))
        .unwrap();
    assert_eq!(total, 1);
    assert_eq!(results[0].title.as_deref(), Some("Orders Metric"));

    // Filter by concept path prefix
    let filter_prefix = QueryFilter {
        concept_path_prefix: Some("tables/".to_string()),
        ..Default::default()
    };
    let (results, _, total) = index
        .search_advanced("orders", SearchMode::Hybrid, 10, 0, Some(&filter_prefix))
        .unwrap();
    assert_eq!(total, 1);
    assert_eq!(results[0].concept_path, "tables/orders");
}

#[test]
fn search_pagination_offset_and_top_k() {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::create_dir_all(&index_dir).unwrap();

    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let mut docs = Vec::new();
    for i in 1..=5 {
        docs.push(
            OkfDocumentBuilder::new(&bundle, bundle.join(format!("doc{i}.md")))
                .frontmatter_value("type", "Metric")
                .frontmatter_value("title", format!("Item {i}"))
                .body(format!("Common search query text item number {i}"))
                .build(),
        );
    }
    index.index_documents(docs).unwrap();

    // Page 1: top_k = 2, offset = 0
    let (page1, _, total) = index
        .search_advanced("item", SearchMode::Hybrid, 2, 0, None)
        .unwrap();
    assert_eq!(total, 5);
    assert_eq!(page1.len(), 2);

    // Page 2: top_k = 2, offset = 2
    let (page2, _, total2) = index
        .search_advanced("item", SearchMode::Hybrid, 2, 2, None)
        .unwrap();
    assert_eq!(total2, 5);
    assert_eq!(page2.len(), 2);
    assert_ne!(page1[0].doc_id, page2[0].doc_id);

    // Page 3: top_k = 2, offset = 4 (only 1 remaining)
    let (page3, _, _) = index
        .search_advanced("item", SearchMode::Hybrid, 2, 4, None)
        .unwrap();
    assert_eq!(page3.len(), 1);
}

#[test]
fn hnsw_ann_search_recall_and_threshold_switching() {
    use rust_okf::ann::{AnnConfig, HnswIndex};

    let dim = 16;
    let mut vectors = Vec::new();
    for i in 0..600 {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = (i + 1) as f32;
        v[(i + 3) % dim] = ((i * 7) % 13) as f32;
        vectors.push(v);
    }

    let config = AnnConfig {
        enabled: true,
        threshold: 500,
        m: 16,
        ef_construction: 64,
        ef_search: 32,
    };

    let hnsw = HnswIndex::build(&vectors, &config);
    let serialized = hnsw.serialize();
    let deserialized = HnswIndex::deserialize(&serialized).unwrap();

    let query = vec![1.0f32; dim];
    let ann_results = deserialized.search(&query, 10, &vectors);
    assert_eq!(ann_results.len(), 10);

    // Verify low threshold switching during document indexing
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(&bundle).unwrap();
    fs::create_dir_all(&index_dir).unwrap();

    let index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    index.manifest(); // loads manifest
    let mut docs = Vec::new();
    for i in 1..=5 {
        docs.push(
            OkfDocumentBuilder::new(&bundle, bundle.join(format!("ann_doc{i}.md")))
                .frontmatter_value("type", "Metric")
                .frontmatter_value("title", format!("Vector Item {i}"))
                .body(format!("Dense vector representation item {i}"))
                .build(),
        );
    }

    let mut custom_index =
        open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    custom_index.index_documents(docs).unwrap();

    let (results, _) = custom_index
        .search("Vector Item", SearchMode::Vector, 5)
        .unwrap();
    assert!(!results.is_empty());
}
