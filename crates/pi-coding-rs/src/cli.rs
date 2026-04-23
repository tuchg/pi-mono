use clap::{Parser, Subcommand};

use crate::logging::{LogFormat, LoggingOptions};

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

    /// Tracing filter override. Falls back to `RUST_LOG` when omitted.
    #[arg(long)]
    pub log_level: Option<String>,

    /// Tracing output format.
    #[arg(long, value_enum)]
    pub log_format: Option<LogFormat>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

impl Cli {
    pub fn logging_options(&self) -> LoggingOptions {
        LoggingOptions {
            log_level: self.log_level.clone(),
            log_format: self.log_format,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print version information
    Version,
    /// List available models
    Models,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_log_options() {
        let cli = Cli::parse_from([
            "pi",
            "--log-level",
            "debug,pi_agent_rs=trace",
            "--log-format",
            "pretty",
        ]);

        assert_eq!(cli.log_level.as_deref(), Some("debug,pi_agent_rs=trace"));
        assert_eq!(cli.log_format, Some(LogFormat::Pretty));
    }

    #[test]
    fn builds_logging_options() {
        let cli = Cli::parse_from(["pi", "--log-format", "full"]);

        assert_eq!(
            cli.logging_options(),
            LoggingOptions {
                log_level: None,
                log_format: Some(LogFormat::Full),
            }
        );
    }
}
