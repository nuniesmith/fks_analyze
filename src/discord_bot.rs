#[cfg(feature = "discord")]
use anyhow::Result;
#[cfg(feature = "discord")]
use serenity::{async_trait, all::{GatewayIntents, Message, Ready, Interaction}, prelude::*};
#[cfg(feature = "discord")]
use std::sync::Once;
#[cfg(feature = "discord")]
use crate::{config::AnalyzeConfig, scan, suggestion, analyze_kind, prompts};
use crate::shell::{self, ScriptOptions};

#[cfg(feature = "discord")]
use once_cell::sync::Lazy;
#[cfg(feature = "discord")]
use std::sync::{Mutex, RwLock};

#[cfg(feature = "discord")]
static SERVICE_DIRS: Lazy<RwLock<Vec<String>>> = Lazy::new(|| RwLock::new(Vec::new()));
#[cfg(feature = "discord")]
static LAST_DIR_PER_CHANNEL: Lazy<Mutex<std::collections::HashMap<u64,String>>> = Lazy::new(|| Mutex::new(std::collections::HashMap::new()));
#[cfg(feature = "discord")]
static LAST_ANALYSIS_DIRS: Lazy<Mutex<std::collections::HashMap<u64, Vec<String>>>> = Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

#[cfg(feature = "discord")]
static HEALTH_ONCE: Once = Once::new();

#[cfg(feature = "discord")]
async fn start_health_server() {
    // Bind simple HTTP responder on SERVICE_PORT (default 4801 for bot) returning {"status":"ok"}
    let port: u16 = std::env::var("SERVICE_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(4801);
    let addr = format!("0.0.0.0:{port}");
    match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => {
            tracing::info!(target="discord_bot", %addr, "health server listening");
            loop {
                match listener.accept().await {
                    Ok((mut sock, _peer)) => {
                        tokio::spawn(async move {
                            use tokio::io::{AsyncReadExt, AsyncWriteExt};
                            let mut buf = [0u8; 512];
                            let _ = sock.read(&mut buf).await; // ignore request contents
                            let body = b"{\"status\":\"ok\"}";
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = sock.write_all(resp.as_bytes()).await;
                            let _ = sock.write_all(body).await;
                        });
                    }
                    Err(e) => {
                        tracing::warn!(target="discord_bot", error=?e, "health accept error");
                        // brief backoff to avoid tight error loop
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    }
                }
            }
        }
        Err(e) => tracing::error!(target="discord_bot", error=?e, %addr, "failed to bind health server"),
    }
}

#[cfg(feature = "discord")]
struct Handler { root: String, cfg: AnalyzeConfig, token: String }

#[cfg(feature = "discord")]
fn sanitize_subdir(base: &str, rel: &str) -> Option<String> {
    use std::path::PathBuf;
    if rel.contains("..") { return None; }
    let candidate = PathBuf::from(base).join(rel);
    let canon_base = std::fs::canonicalize(base).ok()?;
    let canon_cand = std::fs::canonicalize(&candidate).ok()?;
    if canon_cand.starts_with(&canon_base) && canon_cand.is_dir() { Some(canon_cand.display().to_string()) } else { None }
}

#[cfg(feature = "discord")]
#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        tracing::info!(target: "discord_bot", "Connected as {} (registering slash via raw HTTP + prefix)", ready.user.name);
        // Health server now started earlier in run_blocking; keep this as a safety net in case future changes move code.
        if false { HEALTH_ONCE.call_once(|| { tokio::spawn(async { start_health_server().await; }); }); }
        // Fetch application id
        let app_id = match ctx.http.get_current_application_info().await { Ok(info) => info.id, Err(e) => { tracing::error!(?e, "Failed to get application info"); return; } };
        let guild_id_opt = self.cfg.allowed_guild_id.map(serenity::all::GuildId::new);
        // Gather service directories early for slash command choices
        let base_root = std::env::var("FKS_ROOT").ok().unwrap_or_else(|| self.root.clone());
        let mut svc_dirs: Vec<String> = Vec::new();
        if let Ok(read) = std::fs::read_dir(&base_root) {
            svc_dirs = read.filter_map(|e| e.ok()).filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let p = e.path();
                if p.is_dir() && (name.starts_with("fks_") || name == "shared_repos") { Some(name) } else { None }
            }).collect();
            svc_dirs.sort();
        }
        // Keep only first 24 directories to leave room (Discord max choices: 25). Add a root entry.
        let mut dir_choices: Vec<(String,String)> = vec![("root".into(), ".".into())];
        for d in svc_dirs.iter().take(24) { dir_choices.push((d.clone(), d.clone())); }
        // Define command JSON payloads
        #[derive(serde::Serialize)] struct ChoiceJson { name: String, value: String }
        #[derive(serde::Serialize)] struct OptionJson { name: &'static str, description: &'static str, #[serde(rename="type")] kind: u8, required: bool, #[serde(skip_serializing_if="Option::is_none")] choices: Option<Vec<ChoiceJson>> }
        #[derive(serde::Serialize)] struct CommandJson { name: &'static str, description: &'static str, #[serde(skip_serializing_if="Option::is_none")] options: Option<Vec<OptionJson>> }
        let dir_choice_objs: Vec<ChoiceJson> = dir_choices.into_iter().map(|(n,v)| ChoiceJson { name: n, value: v }).collect();
        let commands: Vec<CommandJson> = vec![
            CommandJson { name: "ping", description: "Ping the bot", options: None },
            CommandJson { name: "scan", description: "Scan repository and summarize", options: None },
            CommandJson { name: "suggest", description: "Generate task suggestions", options: None },
            CommandJson { name: "suggest_enrich", description: "Generate LLM-enriched suggestions", options: None },
            CommandJson { name: "file", description: "Fetch file contents", options: Some(vec![ OptionJson { name: "path", description: "Relative file path", kind: 3, required: true, choices: None } ]) },
            CommandJson { name: "analyze_rust", description: "Run Rust analysis (option 3)", options: None },
            CommandJson { name: "analyze", description: "Run analysis for a language group", options: Some(vec![
                OptionJson { name: "kind", description: "all|python|rust|csharp|jsts|mdtxt|shbash|docker (optional; omit to list)", kind: 3, required: false, choices: None },
                OptionJson { name: "dir", description: "Relative directory or comma-separated list (optional)", kind: 3, required: false, choices: Some(dir_choice_objs) },
            ]) },
            CommandJson { name: "dirs", description: "List available service directories", options: None },
            CommandJson { name: "last", description: "Show last analysis output dirs for this channel", options: None },
            CommandJson { name: "prompts", description: "List prompt templates (optional filters)", options: Some(vec![
                OptionJson { name: "category", description: "Filter by category", kind: 3, required: false, choices: None },
                OptionJson { name: "name", description: "Substring filter on name", kind: 3, required: false, choices: None },
            ]) },
            CommandJson { name: "prompt", description: "Show a single prompt (exact name)", options: Some(vec![
                OptionJson { name: "name", description: "Exact prompt name", kind: 3, required: true, choices: None },
                OptionJson { name: "fill", description: "Fill placeholders (yes|no)", kind: 3, required: false, choices: None },
            ]) },
            CommandJson { name: "help", description: "List available commands", options: None },
        ];
        // Bulk upsert (PUT) all commands at once — atomic replacement semantics per Discord API
        if guild_id_opt.is_none() {
            tracing::warn!(target="discord_bot", "No DISCORD_GUILD_ID set: registering GLOBAL commands (may take up to 60m to propagate)");
        }
        let base = if let Some(gid) = guild_id_opt { format!("https://discord.com/api/v10/applications/{app_id}/guilds/{}/commands", gid.get()) } else { format!("https://discord.com/api/v10/applications/{app_id}/commands") };
        let client = reqwest::Client::new();
        let auth_header = format!("Bot {}", self.token);
        match client.put(&base).header("Authorization", &auth_header).json(&commands).send().await {
            Ok(resp) => {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_else(|_| "<body read error>".into());
                if status.is_success() {
                    tracing::info!(target="discord_bot", status=%status, body=%body_text, scope=%(if guild_id_opt.is_some() {"guild"} else {"global"}), "Slash command bulk upsert success");
                } else {
                    if status.as_u16() == 401 { tracing::error!(target="discord_bot", status=%status, body=%body_text, "Slash command bulk upsert failed (401). Ensure token is correct and bot invited with application.commands scope"); } else { tracing::error!(target="discord_bot", status=%status, body=%body_text, "Slash command bulk upsert failed"); }
                }
            }
            Err(e) => tracing::error!(target="discord_bot", ?e, "Slash command bulk upsert HTTP error"),
        }
        // Fetch back the list we see now for diagnostics
        match client.get(&base).header("Authorization", &auth_header).send().await {
            Ok(list_resp) => {
                let status = list_resp.status();
                match list_resp.text().await {
                    Ok(t) => tracing::info!(target="discord_bot", status=%status, commands_json=%t, "Post-registration command listing"),
                    Err(e) => tracing::warn!(target="discord_bot", ?e, status=%status, "Could not read command list body"),
                }
            }
            Err(e) => tracing::error!(target="discord_bot", ?e, "Failed to GET command list"),
        }
        if guild_id_opt.is_some() { tracing::info!("Guild slash commands deployed (visible immediately). If you still don't see them, confirm the bot was invited with 'application.commands' scope."); } else { tracing::info!("Global slash commands registered (client cache may take up to 60m). Consider setting DISCORD_GUILD_ID for faster iteration."); }

        // Prime directory cache once at startup
        let base_root = std::env::var("FKS_ROOT").ok().unwrap_or_else(|| self.root.clone());
        if let Ok(read) = std::fs::read_dir(&base_root) {
            let mut dirs: Vec<String> = read.filter_map(|e| e.ok()).filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let p = e.path();
                if p.is_dir() && (name.starts_with("fks_") || name == "shared_repos") { Some(name) } else { None }
            }).collect();
            dirs.sort();
            if let Ok(mut w) = SERVICE_DIRS.write() { *w = dirs; }
        }
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot { return; }
        if let Some(allowed) = self.cfg.allowed_guild_id { if let Some(gid) = msg.guild_id { if gid.get() != allowed { return; } } else { return; } }
        if let Some(ch) = self.cfg.allowed_channel_id { if msg.channel_id.get() != ch { return; } }
        let content = msg.content.trim();
        let prefix = "!"; if !content.starts_with(prefix) { return; }
        let parts: Vec<&str> = content[prefix.len()..].split_whitespace().collect();
        if parts.is_empty() { return; }
        let cmd = parts[0];
        tracing::info!(target="discord_bot", kind="prefix", user_id=%msg.author.id, command=%cmd, "Prefix command received");
        match cmd {
            "dirs" => {
                // Serve cached list, refresh if empty
                let mut need_refresh = false;
                let _list_cached = {
                    if let Ok(r) = SERVICE_DIRS.read() { if r.is_empty() { need_refresh = true; None } else { Some(r.clone()) } } else { None }
                };
                if need_refresh {
                    let base_root = std::env::var("FKS_ROOT").ok().unwrap_or_else(|| self.root.clone());
                    if let Ok(read) = std::fs::read_dir(&base_root) {
                        let mut dirs: Vec<String> = read.filter_map(|e| e.ok()).filter_map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            let p = e.path();
                            if p.is_dir() && (name.starts_with("fks_") || name == "shared_repos") { Some(name) } else { None }
                        }).collect();
                        dirs.sort();
                        if let Ok(mut w) = SERVICE_DIRS.write() { *w = dirs; }
                    }
                }
                let entries: Vec<String> = SERVICE_DIRS.read().ok().map(|r| r.clone()).unwrap_or_default();
                let list = if entries.is_empty() { "No service directories found".into() } else { entries.join(", ") };
                let _ = msg.reply(&ctx.http, format!("Service dirs: {list}" )).await;
            }
            "help" => { let _ = msg.reply(&ctx.http, "Commands: /ping /scan /suggest /suggest_enrich /file /analyze /analyze_rust /dirs /last /prompts /help (prefix: !ping !scan !suggest !file !analyze <kind> [dir|d1,d2] !analyze_rust !dirs !last !prompts !prompt <name> !help)").await; }
            "prompts" | "prompt" => {
                // !prompts [category] OR !prompt <name_substr>
                let mut category: Option<String> = None;
                let mut name_filter: Option<String> = None;
                if cmd == "prompts" {
                    category = parts.get(1).map(|s| s.to_string());
                } else { // prompt
                    name_filter = parts.get(1).map(|s| s.to_string());
                }
                let cats = prompts::list_categories();
                let mut items: Vec<&prompts::PromptTemplate> = prompts::catalog().prompts.iter().collect();
                if let Some(c) = &category { items.retain(|p| p.category.eq_ignore_ascii_case(c)); }
                if let Some(nf) = &name_filter { let nf_l = nf.to_ascii_lowercase(); items.retain(|p| p.name.to_ascii_lowercase().contains(&nf_l)); }
                // Build output
                let mut out = String::new();
                if category.is_none() && name_filter.is_none() { out.push_str(&format!("Categories: {}\nUse !prompts <Category> or !prompt <name_part>\n", cats.join(", "))); }
                if cmd == "prompt" && items.len() == 1 {
                    // Show full filled prompt
                    let p = prompts::best_match(items[0].name.as_str()).unwrap_or(items[0]);
                    // Context now sourced from cached fill context; previous repo_name and siblings locals removed as redundant.
                    // Build top extensions (shallow scan) for richer context
                    let ctx_fill = prompts::build_fill_context(&self.root, std::time::Duration::from_secs(60));
                    let filled = prompts::fill_template(&p.template, &ctx_fill);
                    out.push_str(&format!("[{}] {}\n{}\n(files: {} primary_lang: {})\n---\n{}", p.category, p.name, p.description, ctx_fill.total_files, ctx_fill.primary_lang, filled));
                } else {
                    for p in items.iter().take(5) { // cap list
                        out.push_str(&format!("[{}] {}: {}\n{}\n---\n", p.category, p.name, p.description, p.template.lines().next().unwrap_or(""))); }
                    if items.len() == 1 { out.push_str("(Use !prompt <exact_start_of_name> for full template)\n"); }
                }
                if items.len() > 5 { out.push_str(&format!("... (+{} more) refine filters", items.len()-5)); }
                let _ = msg.reply(&ctx.http, truncate(&out, self.cfg.max_discord_reply_chars)).await;
            }
            "last" => {
                let list_opt = {
                    if let Ok(m) = LAST_ANALYSIS_DIRS.lock() { m.get(&msg.channel_id.get()).cloned() } else { None }
                };
                let reply_text = if let Some(list) = list_opt { if list.is_empty() { "No analysis output dirs stored".into() } else { format!("Last analysis output dirs: {}", list.join(" | ")) } } else { "No analysis run recorded yet".into() };
                let _ = msg.reply(&ctx.http, reply_text).await;
            }
            "ping" => { let _ = msg.reply(&ctx.http, "pong").await; }
            "scan" => {
                match scan::scan_repo(&self.root, 0) {
                    Ok(summary) => {
                        let top: String = summary.counts_by_ext.iter().take(5).map(|(e,c)| format!("{e}:{c}")).collect::<Vec<_>>().join(", ");
                        let reply = format!("Scanned files: {} (top: {}) root: {}", summary.total_files, top, self.root);
                        let _ = msg.reply(&ctx.http, reply).await;
                        tracing::info!(target="discord_bot", kind="prefix", cmd="scan", total_files=summary.total_files, root=%self.root, "Scan complete");
                    }
                    Err(e) => { let _ = msg.reply(&ctx.http, format!("Scan error: {e}")).await; }
                }
            }
            "suggest" | "suggest+" => {
                let enrich = cmd.ends_with('+') || parts.get(1).map(|s| *s == "enrich").unwrap_or(false);
                match scan::scan_repo(&self.root, 0) {
                    Ok(sum) => {
                        let mut sug = suggestion::generate(&self.root, &sum, None, None);
                        if enrich { if let Ok(client) = crate::llm::maybe_ollama_client() { if let Ok(enriched) = client.enrich_prompt(&sug.prompt_text) { sug.prompt_text = enriched; } } }
                        let _ = msg.reply(&ctx.http, truncate(&sug.prompt_text, self.cfg.max_discord_reply_chars)).await;
                    }
                    Err(e) => { let _ = msg.reply(&ctx.http, format!("Suggest error: {e}")).await; }
                }
            }
            "file" => {
                if parts.len() < 2 { let _ = msg.reply(&ctx.http, "Usage: !file <relative_path>").await; return; }
                let rel = parts[1];
                let path = std::path::Path::new(&self.root).join(rel);
                if !path.exists() { let _ = msg.reply(&ctx.http, "Not found").await; return; }
                match std::fs::read_to_string(&path) {
                    Ok(mut s) => {
                        if s.len() > self.cfg.allow_file_content_bytes { s.truncate(self.cfg.allow_file_content_bytes); s.push_str("\n... (truncated)\n"); }
                        let _ = msg.reply(&ctx.http, format!("```\n{}\n```", truncate(&s, self.cfg.max_discord_reply_chars - 10))).await;
                    }
                    Err(e) => { let _ = msg.reply(&ctx.http, format!("Read error: {e}")).await; }
                }
            }
            "analyze_rust" | "analyze3" | "analyzers" | "arust" => {
                let in_progress_msg = msg.reply(&ctx.http, "Starting Rust analysis (option 3)...").await.ok();
                let root = self.root.clone();
                let max_chars = self.cfg.max_discord_reply_chars;
                let channel = msg.channel_id;
                let http = ctx.http.clone();
                tokio::spawn(async move {
                    let run_res = tokio::task::spawn_blocking(move || {
                        shell::run_analyze_script(ScriptOptions { root: root.clone(), full: false, lint: false, analyze_type: Some("3".into()), output: None, exclude: None, skip_shared: false })
                    }).await;
                    match run_res {
                        Ok(Ok(reports)) => {
                            if let Some(dir) = reports.output_dir.clone() {
                                let summary_path = format!("{}/summary.txt", dir);
                                let content = std::fs::read_to_string(&summary_path).unwrap_or_else(|_| format!("Analysis complete. Reports in {dir}"));
                                let truncated = truncate(&content, max_chars);
                                let fb_note = if reports.used_fallback { " (fallback tmp)" } else { "" };
                                let _ = channel.say(&http, format!("Rust analysis done (dir: {dir}{fb_note}).\n```\n{}\n```", truncated)).await;
                            } else {
                                let _ = channel.say(&http, "Analysis finished but output dir undetected").await;
                            }
                        }
                        Ok(Err(e)) => { let _ = channel.say(&http, format!("Analysis error: {e}")).await; }
                        Err(e) => { let _ = channel.say(&http, format!("Task join error: {e}")).await; }
                    }
                    if let Some(m) = in_progress_msg { let _ = m.delete(&http).await; }
                });
            }
            "analyze" => {
                if parts.len() < 2 { let _ = msg.reply(&ctx.http, "Usage: !analyze <kind> [dir|dir1,dir2]").await; return; }
                let kind = parts[1].to_lowercase();
                let dir_opt = parts.get(2).map(|s| (*s).to_string());
                let mapped_choice = analyze_kind::map_kind(&kind);
                let base_root = std::env::var("FKS_ROOT").ok().unwrap_or_else(|| self.root.clone());
                if let Some(d) = dir_opt.clone() { if let Ok(mut map) = LAST_DIR_PER_CHANNEL.lock() { map.insert(msg.channel_id.get(), d.clone()); } }
                let dirs: Vec<String> = dir_opt.as_ref().map(|d| d.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_else(|| vec![".".into()]);
                let in_progress_msg = msg.reply(&ctx.http, format!("Starting analysis kind={kind} (choice {mapped_choice}) dirs={}", dir_opt.clone().unwrap_or_else(|| ".".into()))).await.ok();
                let max_chars = self.cfg.max_discord_reply_chars;
                let channel = msg.channel_id;
                let http = ctx.http.clone();
                let kind_owned = kind.clone();
                let choice_owned = mapped_choice.to_string();
                let dirs_display = dirs.clone().join(",");
                tokio::spawn(async move {
                    let mut combined = String::new();
                    let mut recorded: Vec<String> = Vec::new();
                    for d in dirs {
                        let target_root = if d == "." { base_root.clone() } else { sanitize_subdir(&base_root, &d).unwrap_or(base_root.clone()) };
                        let label = if d == "." { "(root)".into() } else { d.clone() };
                        let res = tokio::task::spawn_blocking({
                            let choice_owned_inner = choice_owned.clone();
                            let target_root_inner = target_root.clone();
                            move || shell::run_analyze_script(ScriptOptions { root: target_root_inner.clone(), full: false, lint: false, analyze_type: Some(choice_owned_inner), output: None, exclude: None, skip_shared: false })
                        }).await;
                        match res {
                            Ok(Ok(reports)) => {
                                if let Some(dir_out) = reports.output_dir.clone() {
                                    let summary_path = format!("{}/summary.txt", dir_out);
                                    let content = std::fs::read_to_string(&summary_path).unwrap_or_else(|_| format!("Analysis complete. Reports in {dir_out}"));
                                    let truncated = truncate(&content, max_chars.min(2000));
                                    let fb_note = if reports.used_fallback { " (fallback tmp)" } else { "" };
                                    combined.push_str(&format!("Dir {label} -> {dir_out}{fb_note}\n{truncated}\n---\n"));
                                    recorded.push(format!("{label}:{dir_out}{fb_note}"));
                                } else { combined.push_str(&format!("Dir {label}: finished (no output dir)\n---\n")); }
                            }
                            Ok(Err(e)) => combined.push_str(&format!("Dir {label} error: {e}\n---\n")),
                            Err(e) => combined.push_str(&format!("Dir {label} task join error: {e}\n---\n")),
                        }
                    }
                    let truncated_all = truncate(&combined, max_chars);
                    let _ = channel.say(&http, format!("Analysis ({kind_owned}) done (dirs: {dirs_display}).\n```\n{}\n```", truncated_all)).await;
                    if !recorded.is_empty() { if let Ok(mut map) = LAST_ANALYSIS_DIRS.lock() { map.insert(channel.get(), recorded); } }
                    if let Some(m) = in_progress_msg { let _ = m.delete(&http).await; }
                });
            }
            _ => { let _ = msg.reply(&ctx.http, "Commands: !ping !scan !suggest !file <path> !analyze_rust").await; }
        }
    }
    // interaction_create remains below
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let Interaction::Command(command) = interaction else { return; };
        // Allowlist checks
        if let Some(allowed) = self.cfg.allowed_guild_id { if let Some(gid) = command.guild_id { if gid.get() != allowed { tracing::info!(target="discord_bot", reason="guild_mismatch", received=%gid.get(), allowed=%allowed, cmd=%command.data.name, "Skipping interaction"); return; } } else { tracing::info!(target="discord_bot", reason="no_guild", cmd=%command.data.name, "Skipping interaction"); return; } }
        if let Some(ch) = self.cfg.allowed_channel_id { if command.channel_id.get() != ch { tracing::info!(target="discord_bot", reason="channel_mismatch", received=%command.channel_id.get(), allowed=%ch, cmd=%command.data.name, "Skipping interaction"); return; } }
        let name = command.data.name.as_str();
        tracing::info!(target="discord_bot", kind="slash", user_id=%command.user.id, command=%name, "Slash command received");
        // raw HTTP callback: POST interactions/{id}/{token}/callback {type:4,data:{content}}
        async fn raw_reply(http: &serenity::http::Http, id: u64, token: &str, content: &str) {
            #[derive(serde::Serialize)] struct Data<'a> { content: &'a str }
            #[derive(serde::Serialize)] struct Body<'a> { #[serde(rename="type")] t: u8, data: Data<'a> }
            let url = format!("https://discord.com/api/v10/interactions/{id}/{token}/callback");
            let body = Body { t: 4, data: Data { content } };
            let client = reqwest::Client::new();
            let _ = client.post(url).json(&body).send().await; // no auth header needed
            let _ = http;
        }
        // Deferred ack (for long tasks)
        async fn raw_defer(http: &serenity::http::Http, id: u64, token: &str) {
            #[derive(serde::Serialize)] struct Body { #[serde(rename="type")] t: u8 }
            let url = format!("https://discord.com/api/v10/interactions/{id}/{token}/callback");
            let body = Body { t: 5 }; // DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE
            let client = reqwest::Client::new();
            let _ = client.post(url).json(&body).send().await;
            let _ = http;
        }
        async fn raw_followup(app_id: u64, token: &str, content: &str) {
            #[derive(serde::Serialize)] struct Body<'a> { content: &'a str }
            let url = format!("https://discord.com/api/v10/webhooks/{app_id}/{token}");
            let client = reqwest::Client::new();
            let _ = client.post(url).json(&Body { content }).send().await;
        }
        match name {
            "ping" => raw_reply(&ctx.http, command.id.get(), &command.token, "pong").await,
            "scan" => {
                let (msg, total_opt) = match scan::scan_repo(&self.root, 0) { Ok(summary) => {
                    let top: String = summary.counts_by_ext.iter().take(5).map(|(e,c)| format!("{e}:{c}")).collect::<Vec<_>>().join(", ");
                    (format!("Scanned files: {} (top: {top}) root: {}", summary.total_files, self.root), Some(summary.total_files))
                }, Err(e) => (format!("Scan error: {e}"), None) };
                raw_reply(&ctx.http, command.id.get(), &command.token, &msg).await;
                tracing::info!(target="discord_bot", kind="slash", cmd="scan", status="done", total_files=%total_opt.unwrap_or_default());
            }
            "suggest" | "suggest_enrich" => {
                let enrich = name.ends_with("enrich");
                let msg = match scan::scan_repo(&self.root, 0) { Ok(sum) => {
                    let mut sug = suggestion::generate(&self.root, &sum, None, None);
                    if enrich { if let Ok(client) = crate::llm::maybe_ollama_client() { if let Ok(enriched) = client.enrich_prompt(&sug.prompt_text) { sug.prompt_text = enriched; } } }
                    truncate(&sug.prompt_text, self.cfg.max_discord_reply_chars)
                }, Err(e) => format!("Suggest error: {e}") };
                raw_reply(&ctx.http, command.id.get(), &command.token, &msg).await;
                tracing::info!(target="discord_bot", kind="slash", cmd="suggest", enriched=%enrich, status="done");
            }
            "prompt" => {
                let mut name_opt: Option<String> = None; let mut fill_opt: Option<String> = None;
                for opt in &command.data.options { if opt.name == "name" { if let Some(v)=opt.value.as_str(){ name_opt=Some(v.to_string()); } } if opt.name=="fill" { if let Some(v)=opt.value.as_str(){ fill_opt=Some(v.to_string()); } } }
                let Some(name_val) = name_opt else { raw_reply(&ctx.http, command.id.get(), &command.token, "Missing name").await; return; };
                if let Some(p) = prompts::best_match(&name_val) {
                    // Old repo_name/siblings variables removed; using cached fill context instead.
                    let ctx_fill = prompts::build_fill_context(&self.root, std::time::Duration::from_secs(60));
                    let want_fill = fill_opt.as_deref().map(|v| v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("true") || v=="1").unwrap_or(true);
                    let filled = if want_fill { prompts::fill_template(&p.template, &ctx_fill) } else { p.template.clone() };
                    let header = format!("Prompt: {} [{}]\n{}\n(files: {} primary_lang: {})\n---\n", p.name, p.category, p.description, ctx_fill.total_files, ctx_fill.primary_lang);
                    let out = truncate(&format!("{}{}", header, filled), self.cfg.max_discord_reply_chars);
                    raw_reply(&ctx.http, command.id.get(), &command.token, &out).await;
                } else {
                    raw_reply(&ctx.http, command.id.get(), &command.token, "Prompt not found").await;
                }
            }
            "prompts" => {
                // Collect filters
                let mut category: Option<String> = None; let mut name_f: Option<String> = None;
                for opt in &command.data.options { if opt.name == "category" { if let Some(v)=opt.value.as_str(){ category=Some(v.to_string()); } } if opt.name == "name" { if let Some(v)=opt.value.as_str(){ name_f=Some(v.to_string()); } } }
                let cats = prompts::list_categories();
                let mut items: Vec<&prompts::PromptTemplate> = prompts::catalog().prompts.iter().collect();
                if let Some(c)=&category { items.retain(|p| p.category.eq_ignore_ascii_case(c)); }
                if let Some(n)=&name_f { let nlow = n.to_ascii_lowercase(); items.retain(|p| p.name.to_ascii_lowercase().contains(&nlow)); }
                let mut out = String::new();
                if category.is_none() && name_f.is_none() { out.push_str(&format!("Categories: {}\n", cats.join(", "))); }
                for p in items.iter().take(6) { out.push_str(&format!("[{}] {} - {}\n", p.category, p.name, p.description)); }
                if items.len() > 6 { out.push_str(&format!("... (+{} more) narrow filters", items.len()-6)); }
                let truncated = truncate(&out, self.cfg.max_discord_reply_chars);
                raw_reply(&ctx.http, command.id.get(), &command.token, &truncated).await;
            }
            "file" => {
                let mut path_opt: Option<String> = None;
                for opt in &command.data.options { if opt.name == "path" { if let Some(v) = opt.value.as_str() { path_opt = Some(v.to_string()); } } }
                let msg = if let Some(ref rel) = path_opt { let path = std::path::Path::new(&self.root).join(rel); if !path.exists() { "Not found".into() } else { match std::fs::read_to_string(&path) { Ok(mut s) => { if s.len()>self.cfg.allow_file_content_bytes { s.truncate(self.cfg.allow_file_content_bytes); s.push_str("\n... (truncated)\n"); } truncate(&format!("```\n{}\n```", s), self.cfg.max_discord_reply_chars) }, Err(e)=> format!("Read error: {e}") } } } else { "Missing path option".into() };
                raw_reply(&ctx.http, command.id.get(), &command.token, &msg).await;
                tracing::info!(target="discord_bot", kind="slash", cmd="file", path=%path_opt.as_deref().unwrap_or("<none>"), status="done");
            }
            "analyze_rust" => {
                raw_defer(&ctx.http, command.id.get(), &command.token).await; // show thinking state
                tracing::info!(target="discord_bot", kind="slash", cmd="analyze_rust", stage="deferred");
                let root = self.root.clone();
                let max_chars = self.cfg.max_discord_reply_chars;
                let http = ctx.http.clone();
                let follow_token = command.token.clone();
                let allow_internal_fallback = true;
                // Clone root for use in both spawn_blocking closure and later fallback scan
                let root_for_task = root.clone();
                tokio::spawn(async move {
                    let res = tokio::task::spawn_blocking(move || {
                        shell::run_analyze_script(ScriptOptions { root: root_for_task.clone(), full: false, lint: false, analyze_type: Some("3".into()), output: None, exclude: None, skip_shared: false })
                    }).await;
                    let follow = match res {
                        Ok(Ok(rep)) => {
                            if let Some(dir) = rep.output_dir { let summary_path = format!("{}/summary.txt", dir); let content = std::fs::read_to_string(&summary_path).unwrap_or_else(|_| format!("Done. Reports in {dir}")); truncate(&content, max_chars) } else { "Analysis finished (no dir)".into() }
                        }
                        Ok(Err(e)) => {
                            if allow_internal_fallback {
                                // Provide lightweight internal Rust summary fallback
                                match crate::scan::scan_repo(&root, 0) {
                                    Ok(sum) => {
                                        // counts_by_ext is a Vec<(String, usize)>, not a map; search for rs extension
                                        let rust_count = sum.counts_by_ext.iter().find(|(ext, _)| ext == "rs").map(|(_, c)| *c).unwrap_or(0);
                                        format!("Script failed: {e}\nFallback internal scan: total_files={} rust_files={} top_exts={}", sum.total_files, rust_count, sum.counts_by_ext.iter().take(8).map(|(k,v)| format!("{k}:{v}")).collect::<Vec<_>>().join(", "))
                                    }
                                    Err(se) => format!("Analysis error: {e}; fallback scan failed: {se}"),
                                }
                            } else { format!("Analysis error: {e}") }
                        }
                        Err(e) => format!("Join error: {e}"),
                    };
                    let app_id = http.application_id().unwrap_or_default().get();
                    let _ = raw_followup(app_id, &follow_token, &format!("Rust analysis result:\n```\n{}\n```", follow)).await;
                    tracing::info!(target="discord_bot", kind="slash", cmd="analyze_rust", status="done");
                });
            }
            "analyze" => {
                // If no kind provided, immediately reply with list instead of running
                let mut kind: Option<String> = None;
                let mut dir: Option<String> = None;
                for opt in &command.data.options { 
                    if opt.name == "kind" { if let Some(v)= opt.value.as_str(){ kind = Some(v.to_lowercase()); } }
                    if opt.name == "dir" { if let Some(v)= opt.value.as_str(){ dir = Some(v.to_string()); } }
                }
                // If dir omitted and we have last-used for this channel, reuse
                if dir.is_none() { if let Ok(map) = LAST_DIR_PER_CHANNEL.lock() { if let Some(prev) = map.get(&command.channel_id.get()) { dir = Some(prev.clone()); } } }
                if kind.is_none() {
                    let list = format!("{}\nUse /analyze kind:<name> to run.", analyze_kind::ANALYZE_KIND_HELP);
                    raw_reply(&ctx.http, command.id.get(), &command.token, &list).await;
                    return;
                }
                raw_defer(&ctx.http, command.id.get(), &command.token).await;
                let kind_val = kind.unwrap();
                let choice = analyze_kind::map_kind(&kind_val);
                let base_root = std::env::var("FKS_ROOT").ok().unwrap_or_else(|| self.root.clone());
                let dirs: Vec<String> = dir.as_ref().map(|d| d.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_else(|| vec![".".into()]);
                tracing::info!(target="discord_bot", kind="slash", cmd="analyze", selection=%kind_val, mapped=%choice, dirs=?dirs, stage="deferred");
                let max_chars = self.cfg.max_discord_reply_chars;
                let http = ctx.http.clone();
                let follow_token = command.token.clone();
                let choice_owned = choice.to_string();
                let selection_owned = kind_val.clone();
                let dir_disp = dir.clone().unwrap_or_else(|| ".".into());
                tokio::spawn(async move {
                    let mut combined = String::new();
                    let mut recorded: Vec<String> = Vec::new();
                    for d in dirs {
                        let target_root = if d == "." { base_root.clone() } else { sanitize_subdir(&base_root, &d).unwrap_or(base_root.clone()) };
                        let label = if d == "." { "(root)".into() } else { d.clone() };
                        let res = tokio::task::spawn_blocking({
                            let choice_owned_inner = choice_owned.clone();
                            let target_root_inner = target_root.clone();
                            move || shell::run_analyze_script(ScriptOptions { root: target_root_inner.clone(), full: false, lint: false, analyze_type: Some(choice_owned_inner), output: None, exclude: None, skip_shared: false })
                        }).await;
                        match res {
                            Ok(Ok(rep)) => {
                                if let Some(dir_out) = rep.output_dir { let summary_path = format!("{}/summary.txt", dir_out); let content = std::fs::read_to_string(&summary_path).unwrap_or_else(|_| format!("Done. Reports in {dir_out}")); let truncated_part = truncate(&content, max_chars.min(1500)); let fb_note = if rep.used_fallback { " (fallback tmp)" } else { "" }; combined.push_str(&format!("Dir {label} -> {dir_out}{fb_note}\n{truncated_part}\n---\n")); recorded.push(format!("{label}:{dir_out}{fb_note}")); }
                                else { combined.push_str(&format!("Dir {label}: finished (no output dir)\n---\n")); }
                            }
                            Ok(Err(e)) => combined.push_str(&format!("Dir {label} error: {e}\n---\n")),
                            Err(e) => combined.push_str(&format!("Dir {label} task join error: {e}\n---\n")),
                        }
                    }
                    let follow = truncate(&combined, max_chars);
                    let app_id = http.application_id().unwrap_or_default().get();
                    let _ = raw_followup(app_id, &follow_token, &format!("Analysis ({selection_owned}) dir={dir_disp} result:\n```\n{}\n```", follow)).await;
                    if !recorded.is_empty() { if let Ok(mut map) = LAST_ANALYSIS_DIRS.lock() { map.insert(command.channel_id.get(), recorded); } }
                    tracing::info!(target="discord_bot", kind="slash", cmd="analyze", selection=%selection_owned, status="done");
                });
            }
            "last" => {
                let list_opt = { if let Ok(m) = LAST_ANALYSIS_DIRS.lock() { m.get(&command.channel_id.get()).cloned() } else { None } };
                let content = if let Some(list) = list_opt { if list.is_empty() { "No analysis output dirs stored".into() } else { format!("Last analysis output dirs:\n{}", list.join("\n")) } } else { "No analysis run recorded yet".into() };
                raw_reply(&ctx.http, command.id.get(), &command.token, &content).await;
            }
            "dirs" => {
                let entries: Vec<String> = SERVICE_DIRS.read().ok().map(|r| r.clone()).unwrap_or_default();
                let list = if entries.is_empty() { "No service directories found".into() } else { entries.join(", ") };
                raw_reply(&ctx.http, command.id.get(), &command.token, &format!("Service dirs: {list}" )).await;
            }
            "help" => {
                let help_text = format!("Commands:\n/ping - Ping the bot\n/scan - Scan repository\n/suggest - Generate suggestions\n/suggest_enrich - Suggestions with LLM enrichment\n/file path:<rel> - Show file content\n/analyze kind:<all|python|rust|csharp|jsts|mdtxt|shbash|docker> dir:<subdir|d1,d2?> - Analysis (omit kind to list; dir can be comma-separated)\n/prompts [category|name filters] - List prompt templates\n/last - Show last analysis output dirs for this channel\n/analyze_rust - Shortcut for rust (option 3)\n/help - This help\n\n{}", analyze_kind::ANALYZE_KIND_HELP);
                raw_reply(&ctx.http, command.id.get(), &command.token, &help_text).await;
            }
            _ => { raw_reply(&ctx.http, command.id.get(), &command.token, "Unknown command").await; }
        }
    }
}

#[cfg(feature = "discord")]
pub fn run_blocking(root: String) -> Result<()> {
    let cfg = AnalyzeConfig::load()?;
    let token = cfg.require_discord_token()?.to_string();
    let handler_token = token.clone();
    let handler_cfg = cfg.clone();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
    // Start lightweight health server immediately so Docker health checks succeed before Discord gateway ready
    HEALTH_ONCE.call_once(|| { tokio::spawn(async { start_health_server().await; }); });
        let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT | GatewayIntents::DIRECT_MESSAGES;
        let mut client = serenity::Client::builder(token, intents)
            .event_handler(Handler { root, cfg: handler_cfg, token: handler_token })
            .await
            .expect("Err creating client");
        if let Err(why) = client.start().await { tracing::error!(?why, "Client ended"); }
    });
    Ok(())
}

#[cfg(feature = "discord")]
fn truncate(s: &str, max: usize) -> String { if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max]) } }
