use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AnalyzeConfig {
    pub discord_token: Option<String>,
    pub allow_file_content_bytes: usize,
    pub max_discord_reply_chars: usize,
    pub allowed_guild_id: Option<u64>,
    pub allowed_channel_id: Option<u64>,
}

#[allow(dead_code)]
impl AnalyzeConfig {
    pub fn load() -> Result<Self> {
        // Try env first, later support YAML (e.g., fks_analyze.yaml)
        let allow_file_content_bytes = std::env::var("FKS_ALLOW_CONTENT_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(32*1024);
        let max_discord_reply_chars = std::env::var("FKS_MAX_DISCORD_CHARS").ok().and_then(|v| v.parse().ok()).unwrap_or(1800);
        let allowed_guild_id = std::env::var("DISCORD_GUILD_ID").ok().and_then(|v| v.parse().ok());
        let allowed_channel_id = std::env::var("DISCORD_CHANNEL_ID").ok().and_then(|v| v.parse().ok());
        Ok(Self { 
            discord_token: std::env::var("DISCORD_TOKEN").ok(),
            allow_file_content_bytes,
            max_discord_reply_chars,
            allowed_guild_id,
            allowed_channel_id,
        })
    }
    pub fn require_discord_token(&self) -> Result<&str> { self.discord_token.as_deref().context("Missing DISCORD_TOKEN env variable") }
}
