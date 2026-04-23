mod cli;
mod logging;
mod session;

use anyhow::Result;
use clap::Parser;

use cli::Cli;
use logging::init_tracing;

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.logging_options());
    tracing::debug!(?cli.command, print = cli.print, has_prompt = cli.prompt.is_some(), "parsed CLI arguments");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    tracing::info!(?cli.command, "starting pi");

    match cli.command {
        Some(cli::Command::Version) => {
            println!("pi {}", env!("CARGO_PKG_VERSION"));
        }
        Some(cli::Command::Models) => {
            tracing::info!("listing models");
            println!("Model listing not yet implemented");
        }
        None => {
            // Interactive mode — will launch TUI or process stdin
            if let Some(prompt) = cli.prompt {
                tracing::info!(prompt_len = prompt.len(), "running prompt mode");
                println!("Prompt mode: {prompt}");
                // TODO: Run agent loop with the prompt
            } else {
                tracing::info!("running interactive mode");
                println!("Interactive TUI mode not yet implemented");
            }
        }
    }

    Ok(())
}
