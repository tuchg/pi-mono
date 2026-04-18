//! Extension system for the pi agent.
//!
//! Provides the Rust equivalent of the TypeScript extension system from
//! `packages/coding-agent/src/core/extensions/`.
//!
//! # Architecture
//!
//! - **types**: All event types, contexts, and trait definitions
//! - **runner**: Event dispatch and extension lifecycle management
//! - **builder**: API for constructing extensions (the `pi` object)

pub mod builder;
pub mod runner;
pub mod types;

pub use builder::ExtensionBuilder;
pub use runner::{ExtensionContextActions, ExtensionRunner};
pub use types::*;
