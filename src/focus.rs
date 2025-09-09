use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct FocusConfig {
    pub mode: Option<String>, // auto|single|multi
    pub include_patterns: Option<Vec<String>>, // e.g. ["fks_*"]
    pub exclude: Option<Vec<String>>, // e.g. ["shared_repos","backup"]
}

#[allow(dead_code)]
impl FocusConfig { pub fn load() -> Self { Self::default() } }
