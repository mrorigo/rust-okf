# rust-okf

<p align="left">
  <a href="https://crates.io/crates/rust-okf"><img src="https://img.shields.io/crates/v/rust-okf.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/rust-okf"><img src="https://docs.rs/rust-okf/badge.svg" alt="Docs.rs"></a>
  <a href="./Cargo.toml"><img src="https://img.shields.io/badge/rust-1.93%2B-orange.svg" alt="Rust 1.93+"></a>
  <a href="./Cargo.toml"><img src="https://img.shields.io/badge/search-hybrid%20%2B%20vector%20%2B%20bm25-0f766e.svg" alt="Hybrid search"></a>
</p>

`rust-okf` is a Rust-native index and query engine for OKF bundles. It is built for speed, shaped for real data, and tuned for hybrid retrieval without the usual index-engine bloat.

It is built for speed and keeps the core search path deliberately simple:

- FastEmbed for dense embeddings by default
- BM25 for lexical retrieval
- Reciprocal Rank Fusion for hybrid ranking
- immutable on-disk segments with atomic commits
- tombstone-based incremental updates

The goal is not a toy demo. The goal is a small, sharp engine that can index OKF bundles, survive restarts, and answer search queries with low latency.

## At A Glance

- hybrid retrieval over OKF Markdown bundles
- production embeddings via FastEmbed
- atomic manifest-driven persistence
- CLI for indexing and search
- HTTP API with OpenAPI schema

## What it does

- scans OKF bundles from Markdown files with YAML frontmatter
- extracts document metadata and searchable text
- stores hybrid search state on disk
- supports updates and deletes without full rebuilds
- exposes both a CLI and an HTTP API

## Features

- default production embeddings via `fastembed`
- hybrid search with lexical + vector candidate generation
- RRF fusion for ranking
- memory-mapped segment reads
- versioned manifest for safe recovery
- explicit OpenAPI schema for the HTTP API

## Quick Start

```bash
cargo run -- init-config
cargo run -- add ./my-bundle
cargo run -- serve --bind 127.0.0.1:8787
```

## Project layout

- `src/okf.rs` parses and normalizes OKF documents
- `src/bm25.rs` implements lexical scoring
- `src/embedding.rs` wraps FastEmbed and the mock provider
- `src/storage.rs` handles manifests and binary segment files
- `src/index.rs` coordinates indexing, updates, deletes, and search
- `src/api.rs` exposes the HTTP server
- `src/schema.rs` defines request/response DTOs
- `src/openapi.rs` generates the OpenAPI document

## Requirements

- Rust 1.93+ recommended
- network access the first time FastEmbed downloads model weights

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```

## Configuration

`rust-okf` uses a TOML config file, defaulting to `okf.toml`.

Create a default config:

```bash
cargo run -- init-config --config okf.toml
```

Example config:

```toml
[fastembed]
enabled = true
model = "BAAI/bge-small-en-v1.5"

bind = "127.0.0.1:8787"
index = "./okf-index"
```

## CLI

### Initialize config

```bash
cargo run -- init-config
```

### Index a bundle

```bash
cargo run -- add ./my-bundle
```

### Update a bundle

```bash
cargo run -- update ./my-bundle
```

### Delete documents

```bash
cargo run -- delete --doc-id <doc-id>
cargo run -- delete --logical-key <bundle>::<concept-path>
```

### Search

```bash
cargo run -- search "orders completed" --mode hybrid --top-k 10
```

Search modes:

- `lexical`
- `vector`
- `hybrid`

### Run the HTTP API

```bash
cargo run -- serve --bind 127.0.0.1:8787
```

## HTTP API

The server exposes:

- `GET /health`
- `GET /openapi.json`
- `POST /search`
- `POST /documents`
- `POST /documents/update`
- `POST /documents/delete`

### Search request

```json
{
  "query": "orders completed",
  "mode": "hybrid",
  "top_k": 10
}
```

### Document input

```json
{
  "bundle_path": "./my-bundle",
  "file_path": "./my-bundle/tables/orders.md",
  "frontmatter": {
    "type": "Metric",
    "title": "Orders"
  },
  "body": "Orders completed by customers"
}
```

## Storage model

The index is persisted as:

- a versioned manifest
- immutable segment directories
- a custom binary segment file per segment
- memory-mapped reads for segment data

Deletes are represented as tombstones in the manifest. Updates are implemented as delete + reindex under the same logical OKF key.

## Search pipeline

1. Parse the query
2. Embed the query with FastEmbed
3. Score lexical candidates with BM25
4. Score dense candidates with vector similarity
5. Fuse candidate lists with RRF
6. Return ranked results with snippets and score breakdowns

## OKF document model

Each OKF document carries:

- logical key
- bundle path
- concept path
- source file path
- type
- title
- description
- resource
- tags
- timestamp
- body
- searchable text

## Development notes

- The mock embedder still exists for tests and offline experimentation.
- The production path uses FastEmbed by default.
- The current implementation favors clarity and speed of iteration over maximum compression.
- The binary segment format is intentionally explicit so it can evolve without breaking the manifest contract.

## Status

This is an active implementation, not a frozen API.

If you use it as a library, pin versions carefully and expect the storage format and API surface to keep evolving while the engine hardens.

---

If you want to plug this into a larger OKF workflow, the best entry points are:

- [`src/index.rs`](./src/index.rs) for indexing and search
- [`src/api.rs`](./src/api.rs) for HTTP wiring
- [`src/storage.rs`](./src/storage.rs) for the on-disk format
