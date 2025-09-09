mod config;
pub mod scan;
pub mod multi_scan;
mod focus;
mod filter;
pub mod index;
pub mod llm;
mod watch;
mod shell;
pub mod suggestion;
pub mod persist;
pub mod diff;
pub mod analyzer;
pub mod findings_persist;
pub mod pipeline;
pub mod prompts;
#[cfg(feature = "server")]
pub mod server;
// selective test
