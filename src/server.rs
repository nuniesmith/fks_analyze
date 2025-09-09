#[cfg(feature = "server")]
use anyhow::Result;
#[cfg(feature = "server")]
use axum::{routing::get, Router, extract::Query, response::IntoResponse, Json};
#[cfg(feature = "server")]
use std::{net::SocketAddr, sync::Arc};
#[cfg(feature = "server")]
use serde::Deserialize;
#[cfg(feature = "server")]
use crate::{scan, suggestion, diff, llm, index, analyzer, findings_persist, prompts};

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct ScanParams {
    pub content_bytes: Option<usize>,
    pub skip_shared: Option<bool>,
    pub ignore: Option<String>,
}

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct SuggestParams { pub enrich: Option<bool>, pub format: Option<String> }

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct DiffParams { pub from: Option<String>, pub to: Option<String>, pub format: Option<String> }

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct SearchParams { pub query: Option<String>, pub limit: Option<usize> }

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct PromptsParams { pub format: Option<String>, pub category: Option<String>, pub name: Option<String> }

#[cfg(feature = "server")]
#[derive(Deserialize, Debug, Default)]
pub struct PromptParams { pub name: Option<String>, pub fill: Option<bool>, pub format: Option<String> }

#[cfg(feature = "server")]
pub fn build_app(root: String) -> Router {
    let shared_root = Arc::new(root);
    Router::new()
            .route("/version", get(|| async { Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")})) }))
            .route("/stats", get({
                let root = shared_root.clone();
                move || {
                    let root = root.clone();
                    async move {
                        match scan::scan_repo(&root, 0) {
                            Ok(s) => Json(serde_json::json!({
                                "root": s.root,
                                "total_files": s.total_files,
                                "total_size": s.total_size,
                                "counts_by_ext": s.counts_by_ext.iter().take(15).collect::<Vec<_>>()
                            })).into_response(),
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/health", get(|| async { Json(serde_json::json!({"status":"ok"})) }))
            .route("/scan", get({
                let root = shared_root.clone();
                move |q: Query<ScanParams>| {
                    let root = root.clone();
                    async move {
                        let content_bytes = q.content_bytes.unwrap_or(0);
                        let skip_shared = q.skip_shared.unwrap_or(false);
                        let ignore_vec: Vec<String> = q.ignore.as_ref().map(|s| s.split(',').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()).unwrap_or_default();
                        match scan::scan_repo_with_ignore(&root, content_bytes, skip_shared, &ignore_vec) {
                            Ok(summary) => Json(summary).into_response(),
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/suggest", get({
                let root = shared_root.clone();
                move |q: Query<SuggestParams>| {
                    let root = root.clone();
                    async move {
                        match scan::scan_repo(&root, 0) {
                            Ok(scan_sum) => {
                                let diff_opt = diff::latest_diff(&root).ok();
                                let mut sug = suggestion::generate(&root, &scan_sum, diff_opt.as_ref(), None);
                                if q.enrich.unwrap_or(false) { if let Ok(client) = llm::maybe_ollama_client() { if let Ok(enriched) = client.enrich_prompt(&sug.prompt_text) { sug.prompt_text = enriched; } } }
                                let format = q.format.as_deref().unwrap_or("text");
                                if format == "json" { Json(serde_json::json!({"tasks": sug.tasks, "prompt": sug.prompt_text})).into_response() } else { sug.prompt_text.into_response() }
                            }
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/diff", get({
                let root = shared_root.clone();
                move |q: Query<DiffParams>| {
                    let root = root.clone();
                    async move {
                        match diff::compute_from_args(&root, q.from.clone(), q.to.clone()) {
                            Ok(d) => {
                                let format = q.format.as_deref().unwrap_or("text");
                                if format == "json" { Json(d).into_response() } else { d.render_text().into_response() }
                            }
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/search", get({
                let root = shared_root.clone();
                move |q: Query<SearchParams>| {
                    let root = root.clone();
                    async move {
                        let Some(query) = q.query.clone() else { return (axum::http::StatusCode::BAD_REQUEST, "missing query").into_response(); };
                        let limit = q.limit.unwrap_or(20);
                        let idx = match index::open_or_build(&root, 0) { Ok(i)=>i, Err(e)=> return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response() };
                        match idx.search(&query, limit) {
                            Ok(hits) => Json(hits.into_iter().map(|h| serde_json::json!({"path": h.path, "score": h.score, "snippet": h.snippet})).collect::<Vec<_>>()).into_response(),
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/index", get({
                let root = shared_root.clone();
                move || {
                    let root = root.clone();
                    async move {
                        match index::rebuild(&root, 0) {
                            Ok(idx) => Json(serde_json::json!({"status":"ok","files": idx.len()})).into_response(),
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/findings", get({
                let root = shared_root.clone();
                move || {
                    let root = root.clone();
                    async move {
                        // Return latest or generate ephemeral
                        match findings_persist::list_findings(&root) {
                            Ok(list) => {
                                if let Some(p) = list.last() {
                                    match findings_persist::load_findings(p) {
                                        Ok(rep) => Json(rep).into_response(),
                                        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                                    }
                                } else {
                                    // generate quick one
                                    match scan::scan_repo(&root, 2048) {
                                        Ok(s) => { let mut rep = analyzer::default_composite().run(&s); let cfg = analyzer::AnalyzerConfig::from_env(); analyzer::apply_suppressions(&mut rep, &cfg, &root); Json(rep).into_response() },
                                        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                                    }
                                }
                            }
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/focus", get({
                let root = shared_root.clone();
                move || {
                    let root = root.clone();
                    async move {
                        let latest = findings_persist::list_findings(&root).ok();
                        let report = if let Some(l) = latest { if let Some(p) = l.last() { findings_persist::load_findings(p).ok() } else { None } } else { None };
                        let rep = if let Some(r) = report { r } else {
                            match scan::scan_repo(&root, 2048) {
                                Ok(s) => { let mut rep = analyzer::default_composite().run(&s); let cfg = analyzer::AnalyzerConfig::from_env(); analyzer::apply_suppressions(&mut rep, &cfg, &root); rep },
                                Err(e) => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                            }
                        };
                        let tasks = analyzer::top_focus_tasks(&rep, 10);
                        Json(serde_json::json!({"generated_at": rep.meta.get("generated_at"), "tasks": tasks})).into_response()
                    }
                }
            }))
            .route("/trend", get({
                let root = shared_root.clone();
                move || {
                    let root = root.clone();
                    async move {
                        match findings_persist::list_findings(&root) {
                            Ok(paths) => {
                                use analyzer::Severity;
                                let mut rows = Vec::new();
                                for p in paths.iter().rev().take(25).rev() {
                                    if let Ok(rep) = findings_persist::load_findings(p) {
                                        let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
                                        let get = |s: &Severity| rep.counts.get(s).copied().unwrap_or(0);
                                        rows.push(serde_json::json!({
                                            "file": fname,
                                            "info": get(&Severity::Info),
                                            "low": get(&Severity::Low),
                                            "medium": get(&Severity::Medium),
                                            "high": get(&Severity::High),
                                            "critical": get(&Severity::Critical)
                                        }));
                                    }
                                }
                                Json(rows).into_response()
                            }
                            Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                        }
                    }
                }
            }))
            .route("/prompts", get({
                move |q: Query<PromptsParams>| async move {
                    let catalog = prompts::catalog();
                    let mut items: Vec<&prompts::PromptTemplate> = catalog.prompts.iter().collect();
                    if let Some(cat) = &q.category { items.retain(|p| p.category.eq_ignore_ascii_case(cat)); }
                    if let Some(name) = &q.name { items.retain(|p| p.name.to_ascii_lowercase().contains(&name.to_ascii_lowercase())); }
                    let format = q.format.as_deref().unwrap_or("text");
                    match format {
                        "json" => Json(serde_json::json!({
                            "guidelines": catalog.guidelines,
                            "categories": prompts::list_categories(),
                            "count": items.len(),
                            "prompts": items.iter().map(|p| serde_json::json!({
                                "category": p.category,
                                "name": p.name,
                                "description": p.description,
                                "template": p.template,
                            })).collect::<Vec<_>>()
                        })).into_response(),
                        "md" | "markdown" => prompts::to_markdown().into_response(),
                        _ => {
                            let mut out = String::new();
                            out.push_str(&format!("Prompt Guidelines: {}\nCategories: {}\n", catalog.guidelines, prompts::list_categories().join(", ")));
                            for p in items { out.push_str(&format!("\n[{}] {} - {}\n{}\n---\n", p.category, p.name, p.description, p.template)); }
                            out.into_response()
                        }
                    }
                }
            }))
            .route("/prompt", get({
                let root = shared_root.clone();
                move |q: Query<PromptParams>| {
                    let root = root.clone();
                    async move {
                        let Some(name) = q.name.clone() else { return (axum::http::StatusCode::BAD_REQUEST, "missing name").into_response(); };
                        let catalog = prompts::catalog();
                        let Some(p) = prompts::best_match(&name) else { return (axum::http::StatusCode::NOT_FOUND, "not found").into_response(); };
                        let fill = q.fill.unwrap_or(false);
                        let repo_name = std::path::Path::new(&*root).file_name().and_then(|s| s.to_str()).unwrap_or("repo");
                        let mut context_block = String::new();
                        let mut filled: Option<String> = None;
                        let mut total_files_val: usize = 0;
                        let mut primary_lang_val: String = String::new();
                        if fill {
                            use std::time::Duration;
                            let ctx = prompts::build_fill_context(&root, Duration::from_secs(60));
                            let core = prompts::fill_template(&p.template, &ctx);
                            let top_line = ctx.top_exts.iter().take(8).map(|(e,c)| format!("{e}:{c}")).collect::<Vec<_>>().join(", ");
                            context_block = format!("Context: repo={repo_name} files={} primary_lang={} top_exts=[{}]\n", ctx.total_files, ctx.primary_lang, top_line);
                            filled = Some(core);
                            total_files_val = ctx.total_files;
                            primary_lang_val = ctx.primary_lang;
                        }
                        let format = q.format.as_deref().unwrap_or("text");
                        match format {
                            "json" => {
                                Json(serde_json::json!({
                                    "name": p.name,
                                    "category": p.category,
                                    "description": p.description,
                                    "template_raw": p.template,
                                    "filled": filled.as_ref().map(|f| format!("{}{}", context_block, f)),
                                    "guidelines": catalog.guidelines,
                                    "total_files": if filled.is_some() { total_files_val } else { 0 },
                                    "primary_lang": if filled.is_some() { primary_lang_val } else { String::new() },
                                })).into_response()
                            },
                            _ => {
                                let mut out = String::new();
                                if let Some(ref f) = filled { out.push_str(&context_block); out.push_str(&format!("{}\nCategory: {}\n{}\n---\n{}", p.name, p.category, p.description, f)); }
                                else { out.push_str(&format!("{}\nCategory: {}\n{}\n---\n{}", p.name, p.category, p.description, p.template)); }
                                out.into_response()
                            }
                        }
                    }
                }
            }))
}

#[cfg(feature = "server")]
pub fn run_server(root: String, port: u16) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        tracing::info!(%root, %port, "Starting fks_analyze HTTP server");
        let app = build_app(root);

        // Bind
        let addr: SocketAddr = ([0,0,0,0], port).into();
        tracing::info!(?addr, "Listening");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}
