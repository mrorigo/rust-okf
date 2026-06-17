/// Rust guideline compliant 2026-06-17
use clap::{Parser, Subcommand};
use rust_okf::{
    load_bundle, open_index, serve_http, AppConfig, FastEmbedProvider, MockEmbeddingProvider,
    SearchMode,
};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "rust-okf")]
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
    },
    Serve {
        #[arg(long)]
        bind: Option<String>,
    },
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
        Commands::Search { query, mode, top_k } => {
            let mode = match mode.as_str() {
                "lexical" => SearchMode::Lexical,
                "vector" => SearchMode::Vector,
                _ => SearchMode::Hybrid,
            };
            let (results, plan) = index.search(&query, mode, top_k)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
            eprintln!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Commands::Serve { bind } => {
            let bind = bind.unwrap_or(config.bind);
            info!(bind = %bind, "starting http server");
            serve_http(index, bind).await?;
        }
        Commands::InitConfig => {}
    }

    Ok(())
}
