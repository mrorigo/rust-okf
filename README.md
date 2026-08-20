# rust-okf

<p align="left">
  <a href="https://github.com/origo/OKF-Manager/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/origo/OKF-Manager/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="https://github.com/origo/OKF-Manager/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/origo/OKF-Manager/release.yml?label=Release" alt="Release workflow status"></a>
  <a href="https://crates.io/crates/rust-okf"><img src="https://img.shields.io/crates/v/rust-okf.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/rust-okf"><img src="https://docs.rs/rust-okf/badge.svg" alt="Docs.rs"></a>
  <a href="./Cargo.toml"><img src="https://img.shields.io/badge/rust-1.93%2B-orange.svg" alt="Rust 1.93+"></a>
  <a href="./Cargo.toml"><img src="https://img.shields.io/badge/search-hybrid%20%2B%20vector%20%2B%20bm25-0f766e.svg" alt="Hybrid search"></a>
</p>

`rust-okf` is a Rust-native index and query engine for OKF bundles. It is built for speed, shaped for real data, and tuned for hybrid retrieval without the usual index-engine bloat.

> [!NOTE]
> **v0.3.1 Release Highlights**:
> - **Human-Friendly CLI Search**: `okf search` now defaults to concise, readable result summaries instead of dumping the full query plan.
> - **Machine-Readable Search Output**: Use `--json` for compact result JSON and `--explain` for detailed ranking diagnostics on stderr.
> - **Separated Result and Diagnostic Streams**: Combine `--json --explain` safely in scripts without mixing result data and query-plan output.
> - **HNSW ANN Vector Indexing**: Sub-linear $O(\log N)$ vector retrieval via pure Rust Hierarchical Navigable Small World (HNSW) graphs persisted directly in memory-mapped segment files (`OKFSEG05`).
> - **Bounded ANN Retrieval**: Vector search requests bounded per-segment candidate sets and falls back to exact scoring when filters or tombstones would otherwise reduce result completeness.
> - **Metadata & Faceted Filtering**: Filter search queries by document `types`, `tags`, and `concept_path_prefix`.
> - **Pagination & Offset Support**: Paginate search results with `offset` / `--offset` and receive `total_hits` in search responses.
> - **Idempotent Ingestion & Parser Fixes**: Automatic tombstone cleanup on re-indexing, CRLF line ending support, and filtering reserved files (`index.md`, `log.md`).

It is built for speed and keeps the core search path deliberately simple:

- FastEmbed for dense embeddings by default
- HNSW ANN indexing for $O(\log N)$ vector retrieval on large segments
- BM25 for lexical retrieval
- Reciprocal Rank Fusion for hybrid ranking
- immutable on-disk segments with atomic commits
- tombstone-based incremental updates

The goal is not a toy demo. The goal is a small, sharp engine that can index OKF bundles, survive restarts, and answer search queries with low latency.

## At A Glance

- hybrid retrieval over OKF Markdown bundles
- HNSW ANN graph vector indexing stored directly on disk (`OKFSEG05`)
- metadata & faceted filtering by type, tag, and concept path
- paginated query execution with offset & total hit count
- production embeddings via FastEmbed
- atomic manifest-driven persistence
- CLI for indexing and search
- HTTP API with OpenAPI schema
- MCP server over stdio for agent tool integration

## What it does

- scans OKF bundles from Markdown files with YAML frontmatter
- extracts document metadata and searchable text
- stores hybrid search state and HNSW graphs on disk
- supports updates and deletes without full rebuilds
- exposes both a CLI and an HTTP API
- exposes an MCP server for tools-compatible AI clients

## Features

- default production embeddings via `fastembed`
- hybrid search with lexical + vector candidate generation
- HNSW ANN vector search for $O(\log N)$ scale queries
- RRF fusion for ranking
- structured metadata filtering (types, tags, concept path prefix)
- pagination & offset support with `total_hits` count
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
- `src/ann.rs` implements HNSW vector index construction, cosine distance, and graph serialization
- `src/embedding.rs` wraps FastEmbed and the mock provider
- `src/storage.rs` handles manifests, binary segment files (`OKFSEG05`), and HNSW graph persistence
- `src/index.rs` coordinates indexing, updates, deletes, search, and ANN thresholding
- `src/api.rs` exposes the HTTP server
- `src/mcp.rs` exposes the stdio MCP server
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

## Benchmarks

Run the Criterion benchmarks to publish repeatable performance numbers for
indexing, hybrid search, and segment load time:

```bash
cargo bench --bench operations
```

The current benchmark suite measures:

- index build throughput for small, medium, and larger corpus sizes
- hybrid search latency across a prebuilt corpus
- segment reopen and bundle load time

Latest recorded benchmark results:

| Benchmark | Result |
| --- | ---: |
| `index_build/10` | `8.04 ms` |
| `index_build/100` | `12.38 ms` |
| `index_build/500` | `34.09 ms` |
| `hybrid_search/query/orders` | `278.63 us` |
| `hybrid_search/query/revenue` | `278.64 us` |
| `hybrid_search/query/customer activity` | `312.25 us` |
| `segment_load/reopen_index` | `595.42 us` |
| `segment_load/load_bundle` | `15.13 us` |

These numbers come from the current Criterion suite and are meant as a
repeatable baseline, not a fixed performance ceiling. Re-run the benchmark
command above after material changes to indexing, storage, or retrieval.

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

[ann]
enabled = true
threshold = 500
m = 16
ef_construction = 64
ef_search = 32

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
cargo run -- search "orders completed"
cargo run -- search "orders completed" --json
cargo run -- search "orders completed" --explain
cargo run -- search "orders completed" --json --explain
```

By default, the CLI prints a concise human-readable result list. Use `--json` for a compact machine-readable result envelope on stdout. Use `--explain` to print the detailed lexical, vector, and fused query plan on stderr. Combining both flags keeps results and diagnostics in separate streams.

Search options:

- `--mode`: `lexical`, `vector`, `hybrid` (default: `hybrid`)
- `--top-k`: number of results to return (default: `10`)
- `--offset`: pagination offset (default: `0`)
- `--filter-type`: filter by document type (can be specified multiple times)
- `--filter-tag`: filter by tag (can be specified multiple times)
- `--filter-path`: concept path prefix filter
- `--json`: emit compact machine-readable results on stdout
- `--explain`: emit the query execution plan on stderr

### Run the HTTP API

```bash
cargo run -- serve --bind 127.0.0.1:8787
```

### Run as an MCP server

`okf mcp` starts an MCP server over standard input and standard output using the official Rust MCP SDK (`rmcp`). This transport is intended for MCP clients that launch local server processes.

```bash
cargo run -- --config okf.toml mcp
```

When using a development index or offline embeddings:

```bash
cargo run -- --config okf.toml --index ./okf-index --mock-embeddings mcp
```

Configure a local MCP client with the built binary:

```json
{
  "mcpServers": {
    "okf": {
      "command": "/absolute/path/to/okf",
      "args": [
        "--config",
        "/absolute/path/to/okf.toml",
        "mcp"
      ]
    }
  }
}
```

The MCP server exposes these tools:

- `search`: hybrid, lexical, or vector search with pagination and metadata filters
- `index_documents`: index document payloads supplied by the client
- `update_documents`: replace documents using their logical keys
- `delete_documents`: delete by document ID or logical key
- `compact`: compact live index segments
- `index_status`: report index generation, segment count, and tombstone count

MCP protocol messages use stdout, so application logs must remain on stderr. The current MCP transport is stdio-only; streamable HTTP MCP is not enabled by this command.

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
  "top_k": 10,
  "offset": 0,
  "filter": {
    "types": ["Metric"],
    "tags": ["sales"],
    "concept_path_prefix": "metrics/"
  }
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
