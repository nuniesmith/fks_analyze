use anyhow::{Result, Context};
use reqwest::blocking::Client;
use std::time::Duration;
use tracing::debug;

pub struct OllamaClient { pub base: String, pub model: String, pub timeout_secs: u64, http: Client }

pub fn maybe_ollama_client() -> Result<OllamaClient> {
    let base = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "mistral".into());
    let timeout_secs: u64 = std::env::var("OLLAMA_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0 && *v < 600)
        .unwrap_or(30);
    let http = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .with_context(|| "Failed to build reqwest client for Ollama")?;
    let client = OllamaClient { base, model, timeout_secs, http };
    debug!(target: "ollama", base = %client.base, model = %client.model, timeout_secs = client.timeout_secs, "Initialized Ollama client");
    Ok(client)
}

impl OllamaClient {
    pub fn enrich_prompt(&self, text: &str) -> Result<String> {
        #[derive(serde::Serialize)] struct Req<'a>{model:&'a str, prompt:&'a str, stream: bool}
        #[derive(serde::Deserialize)] struct Resp { response: String }
        let r: Resp = self.http.post(format!("{}/api/generate", self.base))
            .json(&Req { model: &self.model, prompt: text, stream:false })
            .send()?.json()?;
        Ok(r.response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn lock() -> &'static Mutex<()> { static LOCK: OnceLock<Mutex<()>> = OnceLock::new(); LOCK.get_or_init(|| Mutex::new(())) }

    fn reset_env() { for k in ["OLLAMA_BASE_URL","OLLAMA_MODEL","OLLAMA_TIMEOUT_SECS"] { std::env::remove_var(k); } }

    #[test]
    fn default_timeout_when_env_missing() {
        let _g = lock().lock().unwrap();
        reset_env();
        let c = maybe_ollama_client().unwrap();
        assert_eq!(c.timeout_secs, 30);
        assert_eq!(c.base, "http://localhost:11434");
        assert_eq!(c.model, "mistral");
    }

    #[test]
    fn parses_custom_timeout() {
    let _g = lock().lock().unwrap();
        reset_env();
        std::env::set_var("OLLAMA_TIMEOUT_SECS", "55");
        let c = maybe_ollama_client().unwrap();
        assert_eq!(c.timeout_secs, 55);
    }

    #[test]
    fn invalid_timeout_falls_back() {
    let _g = lock().lock().unwrap();
        reset_env();
        std::env::set_var("OLLAMA_TIMEOUT_SECS", "0");
        let c = maybe_ollama_client().unwrap();
        assert_eq!(c.timeout_secs, 30); // 0 invalid -> default
        std::env::set_var("OLLAMA_TIMEOUT_SECS", "700");
        let c2 = maybe_ollama_client().unwrap();
        assert_eq!(c2.timeout_secs, 30); // > 600 invalid -> default
    }
}
