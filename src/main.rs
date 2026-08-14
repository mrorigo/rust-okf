/// Rust guideline compliant 2026-06-17
use clap::{Parser, Subcommand};
use rust_okf::query::SearchResult;
use rust_okf::{
    load_bundle, open_index, serve_http, AppConfig, FastEmbedProvider, MockEmbeddingProvider,
    SearchMode,
};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "okf")]
#[command(about = "High-performance OKF bundle index and search engine")]
struct Cli {
    #[arg(long, default_value = "okf.toml")]
    config: PathBuf,

    #[arg(long)]
    index: Option<PathBuf>,

    #[arg(long)]
    mock_embeddings: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    InitConfig,
    Add {
        bundle: PathBuf,
    },
    Update {
        bundle: PathBuf,
    },
    Delete {
        #[arg(long)]
        logical_key: Vec<String>,
        #[arg(long)]
        doc_id: Vec<String>,
    },
    Search {
        query: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long)]
        filter_type: Vec<String>,
        #[arg(long)]
        filter_tag: Vec<String>,
        #[arg(long)]
        filter_path: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        explain: bool,
    },
    Serve {
        #[arg(long)]
        bind: Option<String>,
    },
    Compact,
}

fn print_human_results(
    query: &str,
    mode: SearchMode,
    results: &[SearchResult],
    total_hits: usize,
    offset: usize,
) {
    println!("Search: {query}");
    println!(
        "Mode: {} · {total_hits} results · showing {}",
        mode_name(mode),
        results.len()
    );

    if results.is_empty() {
        println!("No matching documents found.");
        return;
    }

    let first = offset + 1;
    let last = offset + results.len();
    println!("Results {first}-{last} of {total_hits}:\n");
    for (index, result) in results.iter().enumerate() {
        println!(
            "{}. {}",
            offset + index + 1,
            result.title.as_deref().unwrap_or("Untitled")
        );
        println!("   {}", result.concept_path);
        if !result.snippet.trim().is_empty() {
            println!("   {}", compact_snippet(&result.snippet));
        }
        println!("   score: {:.4}\n", result.fused_score);
    }
}

fn mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Lexical => "lexical",
        SearchMode::Vector => "vector",
        SearchMode::Hybrid => "hybrid",
    }
}

fn compact_snippet(snippet: &str) -> String {
    const MAX_LENGTH: usize = 200;
    let compact = snippet.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= MAX_LENGTH {
        return compact;
    }
    let truncated: String = compact.chars().take(MAX_LENGTH - 1).collect();
    format!("{truncated}…")
}

fn provider_from_cli(mock: bool) -> anyhow::Result<Box<dyn rust_okf::EmbeddingProvider>> {
    if mock {
        Ok(Box::new(MockEmbeddingProvider::new(16)))
    } else {
        Ok(Box::new(FastEmbedProvider::new_default()?))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    if matches!(cli.command, Commands::InitConfig) {
        AppConfig::save_default(&cli.config)?;
        info!(config = %cli.config.display(), "wrote default config");
        return Ok(());
    }

    let config = AppConfig::load(&cli.config)?;
    let index_path = cli.index.unwrap_or_else(|| PathBuf::from(&config.index));
    std::fs::create_dir_all(&index_path)?;
    let provider = provider_from_cli(cli.mock_embeddings || !config.fastembed.enabled)?;
    let mut index = open_index(&index_path, provider)?;

    match cli.command {
        Commands::Add { bundle } => {
            let docs = load_bundle(&bundle)?;
            index.index_documents(docs)?;
            info!(bundle = %bundle.display(), "indexed bundle");
        }
        Commands::Update { bundle } => {
            let docs = load_bundle(&bundle)?;
            index.update_documents(docs)?;
            info!(bundle = %bundle.display(), "updated bundle");
        }
        Commands::Delete {
            logical_key,
            doc_id,
        } => {
            if !logical_key.is_empty() {
                index.delete_logical_keys(&logical_key)?;
            }
            if !doc_id.is_empty() {
                index.delete_doc_ids(&doc_id)?;
            }
            info!(logical_keys = ?logical_key, doc_ids = ?doc_id, "applied deletions");
        }
        Commands::Search {
            query,
            mode,
            top_k,
            offset,
            filter_type,
            filter_tag,
            filter_path,
            json,
            explain,
        } => {
            let mode = match mode.as_str() {
                "lexical" => SearchMode::Lexical,
                "vector" => SearchMode::Vector,
                _ => SearchMode::Hybrid,
            };
            let filter =
                if !filter_type.is_empty() || !filter_tag.is_empty() || filter_path.is_some() {
                    Some(rust_okf::QueryFilter {
                        types: filter_type,
                        tags: filter_tag,
                        concept_path_prefix: filter_path,
                    })
                } else {
                    None
                };
            let (results, plan, total_hits) =
                index.search_advanced(&query, mode, top_k, offset, filter.as_ref())?;
            let output = serde_json::json!({
                "results": &results,
                "total_hits": total_hits,
                "offset": offset,
                "top_k": top_k,
            });
            if json {
                println!("{}", serde_json::to_string(&output)?);
            } else {
                print_human_results(&query, mode, &results, total_hits, offset);
            }
            if explain {
                eprintln!("{}", serde_json::to_string_pretty(&plan)?);
            }
        }
        Commands::Serve { bind } => {
            let bind = bind.unwrap_or(config.bind);
            info!(bind = %bind, "starting http server");
            serve_http(index, bind).await?;
        }
        Commands::Compact => {
            index.compact()?;
            info!("compacted index");
        }
        Commands::InitConfig => {}
    }

    Ok(())
}
