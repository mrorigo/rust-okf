use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_okf::{load_bundle, open_index, MockEmbeddingProvider, OkfDocumentBuilder, SearchMode};
use std::fs;

fn make_corpus(dir: &std::path::Path, count: usize) -> Vec<rust_okf::OkfDocument> {
    let bundle = dir.join("bundle");
    fs::create_dir_all(&bundle).unwrap();

    (0..count)
        .map(|i| {
            OkfDocumentBuilder::new(&bundle, bundle.join(format!("doc-{i}.md")))
                .frontmatter_value("type", "Metric")
                .frontmatter_value("title", format!("Document {i}"))
                .frontmatter_value("tags", serde_json::json!(["benchmark", "okf"]))
                .body(format!(
                    "Document {i} contains orders revenue customer activity and benchmark text repeated for search."
                ))
                .build()
        })
        .collect()
}

fn index_build_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build");
    for size in [10usize, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let tmp = tempfile::tempdir().unwrap();
                let index_dir = tmp.path().join("index");
                fs::create_dir_all(&index_dir).unwrap();
                let mut index =
                    open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
                let docs = make_corpus(tmp.path(), size);
                index.index_documents(black_box(docs)).unwrap();
            });
        });
    }
    group.finish();
}

fn hybrid_search_bench(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();
    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let docs = make_corpus(tmp.path(), 200);
    index.index_documents(docs).unwrap();

    let mut group = c.benchmark_group("hybrid_search");
    for query in ["orders", "revenue", "customer activity"] {
        group.bench_with_input(BenchmarkId::new("query", query), &query, |b, &query| {
            b.iter(|| {
                let _ = index
                    .search(black_box(query), SearchMode::Hybrid, 10)
                    .unwrap();
            });
        });
    }
    group.finish();
}

fn segment_load_bench(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let index_dir = tmp.path().join("index");
    fs::create_dir_all(&index_dir).unwrap();
    let mut index = open_index(&index_dir, Box::new(MockEmbeddingProvider::new(16))).unwrap();
    let docs = make_corpus(tmp.path(), 200);
    index.index_documents(docs).unwrap();

    let provider =
        || Box::new(MockEmbeddingProvider::new(16)) as Box<dyn rust_okf::EmbeddingProvider>;
    let mut group = c.benchmark_group("segment_load");
    group.bench_function("reopen_index", |b| {
        b.iter(|| {
            let _ = open_index(&index_dir, provider()).unwrap();
        });
    });
    group.bench_function("load_bundle", |b| {
        b.iter(|| {
            let _ = load_bundle(&tmp.path().join("bundle")).unwrap();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    index_build_bench,
    hybrid_search_bench,
    segment_load_bench
);
criterion_main!(benches);
