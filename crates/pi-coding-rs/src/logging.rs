use tracing_subscriber::EnvFilter;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum LogFormat {
    Compact,
    Full,
    Pretty,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoggingOptions {
    pub log_level: Option<String>,
    pub log_format: Option<LogFormat>,
}

impl LoggingOptions {
    pub fn env_filter(&self) -> EnvFilter {
        match self.log_level.as_deref() {
            Some(level) if !level.trim().is_empty() => {
                EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::from_default_env())
            }
            _ => EnvFilter::from_default_env(),
        }
    }
}

pub fn init_tracing(options: &LoggingOptions) {
    let builder = tracing_subscriber::fmt().with_env_filter(options.env_filter());

    match options.log_format.unwrap_or(LogFormat::Compact) {
        LogFormat::Compact => {
            let _ = builder.compact().try_init();
        }
        LogFormat::Full => {
            let _ = builder.try_init();
        }
        LogFormat::Pretty => {
            let _ = builder.pretty().try_init();
        }
    }
}
