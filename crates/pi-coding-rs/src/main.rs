mod cli;
mod session;
mod tools;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use cli::Cli;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    match cli.command {
        Some(cli::Command::Version) => {
            println!("pi {}", env!("CARGO_PKG_VERSION"));
        }
        Some(cli::Command::Models) => {
            println!("Model listing not yet implemented");
        }
        None => {
            // Interactive mode — will launch TUI or process stdin
            if let Some(prompt) = cli.prompt {
                println!("Prompt mode: {prompt}");
                // TODO: Run agent loop with the prompt
            } else {
                println!("Interactive TUI mode not yet implemented");
            }
        }
    }

    Ok(())
}
