use clap::{Parser, Subcommand};

/// pi — interactive coding agent powered by LLMs
#[derive(Debug, Parser)]
#[command(name = "pi", version, about)]
pub struct Cli {
    /// Prompt to send (non-interactive mode)
    pub prompt: Option<String>,

    /// Model to use (provider/model-id)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Working directory
    #[arg(short = 'd', long)]
    pub dir: Option<String>,

    /// Resume a previous session by ID
    #[arg(long)]
    pub resume: Option<String>,

    /// Print mode — output to stdout and exit
    #[arg(long)]
    pub print: bool,

    /// Maximum number of turns
    #[arg(long)]
    pub max_turns: Option<u32>,

    /// Thinking level (off, minimal, low, medium, high, xhigh)
    #[arg(long)]
    pub thinking: Option<String>,

    /// System prompt override
    #[arg(long)]
    pub system_prompt: Option<String>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print version information
    Version,
    /// List available models
    Models,
}
