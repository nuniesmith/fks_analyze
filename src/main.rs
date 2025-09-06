mod config;
mod scan;
mod multi_scan;
mod focus;
mod filter;
mod index;
mod llm;
mod watch;
mod shell;
mod suggestion;
mod persist;
mod diff;
mod analyze_kind;
mod findings_persist;
mod pipeline;
mod analyzer;
mod autofix;
mod report;
mod prompts; // structured prompt catalog
// Helper for fail_on gating
fn evaluate_fail_on(root: &str, threshold: analyzer::Severity, min_sev: Option<analyzer::Severity>, kinds: &[String]) -> anyhow::Result<bool> {
    // Try existing findings snapshot first
    let mut rep = if let Ok(fpaths) = crate::findings_persist::list_findings(root) {
        if let Some(last) = fpaths.last() { crate::findings_persist::load_findings(last)? } else {
            let sc = scan::scan_repo(root, 0)?; let mut r = analyzer::default_composite().run(&sc); let cfg = analyzer::AnalyzerConfig::from_env(); analyzer::apply_suppressions(&mut r, &cfg, root); r
        }
    } else {
        let sc = scan::scan_repo(root, 0)?; let mut r = analyzer::default_composite().run(&sc); let cfg = analyzer::AnalyzerConfig::from_env(); analyzer::apply_suppressions(&mut r, &cfg, root); r
    };
    if let Some(ms) = min_sev { analyzer::filter_min_severity(&mut rep, ms); }
    if !kinds.is_empty() { let allowed: Vec<String> = kinds.iter().map(|s| s.to_ascii_lowercase()).collect(); rep.findings.retain(|f| f.kind.as_ref().map(|k| allowed.contains(&format!("{:?}", k).to_ascii_lowercase())).unwrap_or(false)); }
    Ok(rep.findings.iter().any(|f| f.severity >= threshold))
}

// Return true if the count for the specified severity (exact) increased vs previous findings snapshot.
fn evaluate_fail_on_delta(root: &str, threshold: analyzer::Severity) -> anyhow::Result<bool> {
    if let Ok(mut list) = crate::findings_persist::list_findings(root) {
        if list.len() < 2 { return Ok(false); }
        list.sort();
        let prev = crate::findings_persist::load_findings(&list[list.len()-2])?;
        let curr = crate::findings_persist::load_findings(&list[list.len()-1])?;
        let a = prev.counts.get(&threshold).copied().unwrap_or(0);
        let b = curr.counts.get(&threshold).copied().unwrap_or(0);
        return Ok(b > a);
    }
    Ok(false)
}
#[cfg(feature = "semantic")] mod semantic;
#[cfg(feature = "discord")] mod discord_bot;
#[cfg(feature = "server")] mod server;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about = "FKS Analyze Service: scan, analyze and suggest next dev prompts", long_about=None)]
struct Cli {
    /// Base directory to operate on (defaults to repo root / current dir)
    #[arg(global=true, long)]
    root: Option<String>,

    /// Increase verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Scan the codebase and output a JSON summary
    Scan {
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<String>,
        /// Include file content up to N bytes (0 = none)
        #[arg(long, default_value_t=0)]
        content_bytes: usize,
        /// Save snapshot into .fks_analyze/history
    #[arg(long)] save: bool,
    /// Gzip snapshot when saving (overrides env)
    #[arg(long)] compress: Option<bool>,
    /// Keep only latest N snapshots after saving (override env)
    #[arg(long)] retention: Option<usize>,
    /// Prune snapshots older than N days (override env)
    #[arg(long)] retention_days: Option<i64>,
        /// Skip shared/ subdirectory within each repo (no effect for root scan if root is itself a repo)
    #[arg(long)] skip_shared: bool,
    /// Ignore glob patterns (comma separated)
    #[arg(long)] ignore: Option<String>,
    },
    /// Scan all sibling fks_* repos (auto-detect monorepo root) and output multi summary
    ScanAll {
        #[arg(short, long)] output: Option<String>,
        #[arg(long, default_value_t=0)] content_bytes: usize,
        #[arg(long)] skip_shared: bool,
        /// Include shared_repos aggregate
        #[arg(long)] include_shared_repos: bool,
    /// Ignore glob patterns (comma separated)
    #[arg(long)] ignore: Option<String>,
    /// Save individual repo scan snapshots
    #[arg(long)] save: bool,
    /// Compress saved snapshots
    #[arg(long)] compress: Option<bool>,
    /// Retention (count) for snapshots
    #[arg(long)] retention: Option<usize>,
    /// Retention days for snapshots
    #[arg(long)] retention_days: Option<i64>,
    },
    /// Run the shell analysis script and capture its reports
    Analyze {
        /// Pass --full to script
        #[arg(long)] full: bool,
        /// Pass --lint to script
        #[arg(long)] lint: bool,
        /// Analysis type (maps to script interactive choice). If omitted script will be interactive
        #[arg(long)] analyze_type: Option<String>,
        /// Directory to store reports (script --output)
        #[arg(long)] output: Option<String>,
        /// Exclude directories (comma separated)
        #[arg(long)] exclude: Option<String>,
        /// Skip shared directory
        #[arg(long)] skip_shared: bool,
    },
    /// End-to-end Rust pipeline (scan -> analyze -> suggestions) replacing legacy shell
    Pipeline {
        /// Include up to N bytes per file in scan
        #[arg(long, default_value_t=2048)] content_bytes: usize,
        /// Save scan snapshot
        #[arg(long)] save_scan: bool,
        /// Save findings snapshot
    #[arg(long)] save_findings: bool,
    /// Exit with code 2 if any finding at or above severity exists (info|low|medium|high|critical)
    #[arg(long)] fail_on: Option<String>,
    /// Exit with code 2 if count for severity increased vs previous findings snapshot
    #[arg(long)] fail_on_delta: Option<String>,
    /// Output format text|json
    #[arg(long, default_value="text")] format: String,
    },
    /// Produce task suggestions & AI prompt text from current state
    Suggest {
        /// Use previously generated scan JSON instead of re-scanning
        #[arg(long)] scan_json: Option<String>,
        /// Output format: text|json
        #[arg(long, default_value="text")] format: String,
    /// Enrich suggestions using configured LLM (env: OLLAMA_MODEL / OLLAMA_BASE_URL)
    #[arg(long)] enrich: bool,
    /// Include analyzer findings
    #[arg(long)] with_findings: bool,
    },
    /// Run analyzers & output structured findings
    Findings {
        /// Use existing scan JSON (otherwise scan fresh, no content snippets)
        #[arg(long)] scan_json: Option<String>,
        /// Output file path (optional)
        #[arg(short, long)] output: Option<String>,
        /// Pretty JSON
        #[arg(long)] pretty: bool,
        /// Save snapshot of findings
        #[arg(long)] save: bool,
    /// Exit with non-zero code if any finding at or above this severity exists (info|low|medium|high|critical)
    #[arg(long)] fail_on: Option<String>,
    /// Exit with code 2 if count for severity (or higher) increased vs previous snapshot
    #[arg(long)] fail_on_delta: Option<String>,
    },
    /// Show trend of findings severities over recent snapshots
    Trend {
        /// Limit to last N snapshots
        #[arg(long, default_value_t=20)] limit: usize,
        /// Output format json|text
        #[arg(long, default_value="text")] format: String,
    },
    /// Show top focus tasks from latest findings (or generate if missing)
    Focus {
        /// Number of tasks
        #[arg(long, default_value_t=10)] limit: usize,
        /// If no saved findings, generate (scan snippet bytes)
        #[arg(long, default_value_t=2048)] content_bytes: usize,
    /// Output format text|json
    #[arg(long, default_value="text")] format: String,
    },
    /// Show diff between two snapshots (defaults: latest vs previous)
    Diff {
        /// From snapshot filename (optional)
        #[arg(long)] from: Option<String>,
        /// To snapshot filename (optional)
        #[arg(long)] to: Option<String>,
        /// Output format text|json
    #[arg(long, default_value="text")] format: String,
    /// Include naive unified diffs (requires current workspace files)
    #[arg(long, default_value_t=false)] unified: bool,
    /// Max file size (bytes) to attempt unified diff
    #[arg(long, default_value_t=200_000)] unified_max: usize,
    /// Context lines for unified diff hunks
    #[arg(long, default_value_t=3)] unified_context: usize,
    },
    /// Full text search (requires prior index build)
    Search { #[arg(long)] query: String, #[arg(long, default_value_t=20)] limit: usize },
    /// Build (or rebuild) index
    Index { #[arg(long, default_value_t=0)] content_bytes: usize },
    /// Generate Markdown dashboard report
    Report { #[arg(long, default_value_t=10)] focus_limit: usize, #[arg(long, default_value_t=12)] trend: usize, #[arg(long, default_value_t=2048)] content_bytes: usize, #[arg(short, long)] output: Option<String>, #[arg(long)] min_severity: Option<String>, #[arg(long)] kinds: Option<String>, #[arg(long, default_value="md")] format: String, /// Suppress diff section
    #[arg(long, default_value_t=false)] no_diff: bool, /// Exit with code 2 if any finding at or above severity exists
    #[arg(long)] fail_on: Option<String>, /// Exit with code 2 if severity count increased vs previous findings snapshot
    #[arg(long)] fail_on_delta: Option<String> },
    /// Plan / apply simple automated fixes
    AutoFix { #[arg(long, default_value_t=false)] apply: bool, #[arg(long, default_value_t=false)] dry_run: bool, #[arg(long, default_value_t=0)] content_bytes: usize, #[arg(long)] plan_json: Option<String>, #[arg(long)] diff_out: Option<String> },
    /// Build semantic index (tantivy, feature=semantic)
    #[cfg(feature = "semantic")]
    SemanticBuild { #[arg(long, default_value_t=4096)] content_bytes: usize, #[arg(long)] force: bool },
    /// Semantic search (tantivy, feature=semantic)
    #[cfg(feature = "semantic")]
    SemanticSearch { #[arg(long)] query: String, #[arg(long, default_value_t=10)] limit: usize },
    /// File system watch mode (prints diffs on change)
    Watch { #[arg(long, default_value_t=0)] content_bytes: usize },
    /// Run HTTP server (feature = server)
    #[cfg(feature = "server")]
    Serve { #[arg(long)] port: Option<u16> },
    /// Run Discord bot (feature = discord)
    #[cfg(feature = "discord")] 
    Bot {},
    /// Prune stored scan snapshots by count and/or age
    Prune {
        /// Keep only latest N snapshots (0 = no count pruning)
        #[arg(long, default_value_t=0)] keep: usize,
        /// Remove snapshots older than N days (0 = no age pruning)
        #[arg(long, default_value_t=0)] max_age_days: i64,
        /// Show what would be removed without deleting
        #[arg(long, default_value_t=false)] dry_run: bool,
    },
    /// Show snapshot storage statistics
    Stats { #[arg(long, default_value="text")] format: String, #[arg(long, default_value_t=0)] limit: usize, #[arg(long)] warn_ratio_increase: Option<f64> },
    /// List available structured AI prompt templates
    Prompts { #[arg(long)] category: Option<String>, #[arg(long)] name: Option<String>, #[arg(long, default_value="text")] format: String },
    /// Show a single prompt template by exact name
    Prompt { #[arg(long)] name: String, #[arg(long, default_value_t=false)] fill: bool, #[arg(long, default_value="text")] format: String },
}

fn init_tracing(verbosity: u8) {
    let base = match verbosity { 0 => "info", 1 => "debug", _ => "trace" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(base));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    let root = cli.root.or_else(|| std::env::var("FKS_ROOT").ok()).unwrap_or_else(|| std::env::current_dir().unwrap().display().to_string());
    match cli.command {
    Commands::Scan { output, content_bytes, save, skip_shared, ignore, compress, retention, retention_days } => {
            let ignore_vec: Vec<String> = ignore.as_ref().map(|s| s.split(',').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()).unwrap_or_default();
            let summary = if ignore_vec.is_empty() { scan::scan_repo_opts(&root, content_bytes, skip_shared)? } else { scan::scan_repo_with_ignore(&root, content_bytes, skip_shared, &ignore_vec)? };
            if save { persist::save_snapshot_with(&root, &summary, &persist::SnapshotSaveOptions { compress, retention, retention_days, ..Default::default() })?; }
            if let Some(path) = output { std::fs::write(path, serde_json::to_string_pretty(&summary)?)?; }
            else { println!("{}", serde_json::to_string_pretty(&summary)?); }
    },
    Commands::ScanAll { output, content_bytes, skip_shared, include_shared_repos, ignore, save, compress, retention, retention_days } => {
                let ignore_vec: Vec<String> = ignore.as_ref().map(|s| s.split(',').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()).unwrap_or_default();
                let multi = multi_scan::scan_all(&root, content_bytes, skip_shared, include_shared_repos, &ignore_vec)?;
                if save {
                    use rayon::prelude::*;
                    multi.repos.par_iter().for_each(|repo| {
                        let opts = persist::SnapshotSaveOptions { compress, retention, retention_days, ..Default::default() };
                        let _ = persist::save_snapshot_with(&repo.path, &repo.scan, &opts);
                    });
                }
            if let Some(path) = output { std::fs::write(path, serde_json::to_string_pretty(&multi)?)?; }
            else { println!("{}", serde_json::to_string_pretty(&multi)?); }
    },
    Commands::Analyze { full, lint, analyze_type, output, exclude, skip_shared } => {
            // Map friendly aliases to numeric script choices if provided; show help if omitted
            let analyze_type = analyze_type.map(|v| analyze_kind::map_kind(v));
            if analyze_type.is_none() {
                println!("{}\n(Use --analyze-type <kind|number> to skip interactive menu)", analyze_kind::ANALYZE_KIND_HELP);
            }
            let reports = shell::run_analyze_script(shell::ScriptOptions {
                root: root.clone(), full, lint, analyze_type, output, exclude, skip_shared
            })?;
            println!("Analysis complete: {:?}", reports);
    },
    Commands::Pipeline { content_bytes, save_scan, save_findings, fail_on, fail_on_delta, format } => {
            let out = pipeline::run_pipeline(pipeline::PipelineOptions { root: root.clone(), content_bytes, save_findings, save_scan })?;
            match format.as_str() {
                "json" => {
                    let json = serde_json::json!({
                        "root": root,
                        "snapshot_saved": out.snapshot_saved,
                        "findings_saved": out.findings_saved,
                        "task_count": out.tasks.len(),
                        "tasks": out.tasks,
                        "counts": out.counts,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("Pipeline complete. Tasks ({}):", out.tasks.len());
                    for (i,t) in out.tasks.iter().enumerate() { println!("{}. {}", i+1, t); }
                    if let Some(p) = out.snapshot_saved { println!("Saved scan: {p}"); }
                    if let Some(f) = out.findings_saved { println!("Saved findings: {f}"); }
                }
            }
            if let Some(sev_s) = fail_on { if let Some(threshold) = analyzer::parse_severity(&sev_s) {
                // Determine if any severity >= threshold present using counts
                use analyzer::Severity;
                let order = [Severity::Info, Severity::Low, Severity::Medium, Severity::High, Severity::Critical];
                let mut triggered = false;
                for s in order.iter() { if *s >= threshold { if out.counts.get(s).copied().unwrap_or(0) > 0 { triggered = true; break; } } }
                if triggered { eprintln!("fail_on triggered at {:?}", threshold); std::process::exit(2); }
            } else { eprintln!("Unknown --fail-on severity: {}", sev_s); } }
            if let Some(delta_s) = fail_on_delta { if let Some(threshold) = analyzer::parse_severity(&delta_s) { if evaluate_fail_on_delta(&root, threshold.clone())? { eprintln!("fail_on_delta triggered at {:?}", threshold); std::process::exit(2); } } else { eprintln!("Unknown --fail-on-delta severity: {}", delta_s); } }
    },
    Commands::Suggest { scan_json, format, enrich, with_findings } => {
            let scan_data = if let Some(p) = scan_json { 
                serde_json::from_slice(&std::fs::read(p)?)?
            } else { scan::scan_repo(&root, 0)? };
            // Try to load diff of latest two snapshots for enriched suggestions
            let diff_opt = diff::latest_diff(&root).ok();
            let findings_opt = if with_findings { 
                let comp = analyzer::default_composite();
                let mut report = comp.run(&scan_data);
                let cfg = analyzer::AnalyzerConfig::from_env();
                analyzer::apply_suppressions(&mut report, &cfg, &root);
                Some(report)
            } else { None };
            let mut suggestions = suggestion::generate(&root, &scan_data, diff_opt.as_ref(), findings_opt.as_ref());
            if enrich { if let Ok(client) = llm::maybe_ollama_client() { if let Ok(enriched) = client.enrich_prompt(&suggestions.prompt_text) { suggestions.prompt_text = enriched; } } }
            match format.as_str() { 
                "json" => println!("{}", serde_json::to_string_pretty(&suggestions)?),
                _ => println!("{}", suggestions.prompt_text)
            }
    },
    Commands::Findings { scan_json, output, pretty, save, fail_on, fail_on_delta } => {
            let scan_data = if let Some(p) = scan_json { serde_json::from_slice(&std::fs::read(p)?)? } else { scan::scan_repo(&root, 4096)? };
            let comp = analyzer::default_composite();
            let mut report = comp.run(&scan_data);
            let cfg = analyzer::AnalyzerConfig::from_env();
            analyzer::apply_suppressions(&mut report, &cfg, &root);
            if let Some(min) = cfg.min_severity.clone() { analyzer::filter_min_severity(&mut report, min); }
            if save { let path = crate::findings_persist::save_findings(&root, &report)?; println!("Saved findings -> {}", path.display()); }
            let serialized = if pretty { serde_json::to_string_pretty(&report)? } else { serde_json::to_string(&report)? };
            if let Some(out) = output { std::fs::write(out, serialized)?; } else { println!("{}", serialized); }
            if let Some(sev_s) = fail_on { if let Some(threshold) = analyzer::parse_severity(&sev_s) { if report.findings.iter().any(|f| f.severity >= threshold) { eprintln!("fail_on triggered at {:?}", threshold); std::process::exit(2); } } else { eprintln!("Unknown --fail-on severity: {}", sev_s); } }
            if let Some(delta_s) = fail_on_delta { match analyzer::parse_severity(&delta_s) { Some(th) => if evaluate_fail_on_delta(&root, th.clone())? { eprintln!("fail_on_delta triggered at {:?}", th); std::process::exit(2); }, None => eprintln!("Unknown --fail-on-delta severity: {}", delta_s) } }
    },
    Commands::Trend { limit, format } => {
            let paths = crate::findings_persist::list_findings(&root)?;
            let slice = if paths.len() > limit { &paths[paths.len()-limit..] } else { &paths[..] };
            let mut rows: Vec<(String, usize, usize, usize, usize, usize)> = Vec::new();
            use crate::analyzer::Severity;
            for p in slice {
                if let Ok(rep) = crate::findings_persist::load_findings(p) {
                    let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
                    let get = |s: &Severity| rep.counts.get(s).copied().unwrap_or(0);
                    rows.push((fname, get(&Severity::Info), get(&Severity::Low), get(&Severity::Medium), get(&Severity::High), get(&Severity::Critical)));
                }
            }
            match format.as_str() {
                "json" => {
                    let json_rows: Vec<_> = rows.iter().map(|r| serde_json::json!({"file":r.0, "info":r.1, "low":r.2, "medium":r.3, "high":r.4, "critical":r.5})).collect();
                    println!("{}", serde_json::to_string_pretty(&json_rows)?);
                }
                _ => {
                    println!("Findings Trend (last {}):", rows.len());
                    println!("FILE	INFO	LOW	MED	HIGH	CRIT");
                    for r in rows { println!("{}	{}	{}	{}	{}	{}", r.0, r.1, r.2, r.3, r.4, r.5); }
                }
            }
    },
    Commands::Focus { limit, content_bytes, format } => {
            // Load latest findings or create new
            let latest = crate::findings_persist::list_findings(&root)?;
            use crate::analyzer;
            let report = if let Some(path) = latest.last() { crate::findings_persist::load_findings(path)? } else {
                let scan_data = scan::scan_repo(&root, content_bytes)?;
                analyzer::default_composite().run(&scan_data)
            };
            let tasks = analyzer::top_focus_tasks(&report, limit);
            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "generated_at": report.meta.get("generated_at"),
                        "limit": limit,
                        "tasks": tasks,
                    }))?);
                }
                _ => {
                    println!("Focus Tasks (top {}):", tasks.len());
                    for (i,t) in tasks.iter().enumerate() { println!("{}. {}", i+1, t); }
                }
            }
    },
    Commands::Diff { from, to, format, unified, unified_max, unified_context } => {
            let d = if unified {
                if let Ok((diff_res, from_scan, to_scan)) = diff::compute_with_scans_from_args(&root, from.clone(), to.clone()) {
            diff::enrich_with_unified_snippets(diff_res, &from_scan, &to_scan, unified_max, unified_context)
                } else { diff::compute_from_args(&root, from, to)? }
            } else { diff::compute_from_args(&root, from, to)? };
            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&d)?),
                _ => println!("{}", d.render_text())
            }
    },
    Commands::Search { query, limit } => {
            let idx = index::open_or_build(&root, 0)?; // open existing index
            let results = idx.search(&query, limit)?;
            for r in results { println!("{}:{} score={}", r.path, r.snippet.unwrap_or_default(), r.score); }
    },
    Commands::Index { content_bytes } => {
            let idx = index::open_or_build(&root, content_bytes)?; println!("Index built at {}", idx.dir.display());
    },
    Commands::Report { focus_limit, trend, content_bytes, output, min_severity, kinds, format, no_diff, fail_on, fail_on_delta } => {
            let min_sev = min_severity.and_then(|s| analyzer::parse_severity(&s));
            let kind_vec = kinds.map(|s| s.split(',').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()).unwrap_or_else(|| Vec::new());
            let fmt = match format.as_str() { "json" => report::ReportFormat::Json, "html" => report::ReportFormat::Html, "csv" => report::ReportFormat::Csv, _ => report::ReportFormat::Markdown };
            let out_s = report::generate_report(report::ReportOptions { root: root.clone(), limit: focus_limit, trend, content_bytes, min_severity: min_sev.clone(), kinds: kind_vec.clone(), format: fmt, show_diff: !no_diff })?;
            if let Some(path) = output { std::fs::write(path, out_s)?; } else { println!("{}", out_s); }
            if let Some(sev_s) = fail_on { if let Some(threshold) = analyzer::parse_severity(&sev_s) { if evaluate_fail_on(&root, threshold.clone(), min_sev, &kind_vec)? { eprintln!("fail_on triggered at {:?}", threshold); std::process::exit(2); } } else { eprintln!("Unknown --fail-on severity: {}", sev_s); } }
            if let Some(delta_s) = fail_on_delta { match analyzer::parse_severity(&delta_s) { Some(th) => if evaluate_fail_on_delta(&root, th.clone())? { eprintln!("fail_on_delta triggered at {:?}", th); std::process::exit(2); }, None => eprintln!("Unknown --fail-on-delta severity: {}", delta_s) } }
    },
    Commands::AutoFix { apply, dry_run, content_bytes, plan_json, diff_out } => {
            // Build findings (fresh or cached) with content for context if requested
            let scan_data = scan::scan_repo(&root, content_bytes)?;
            let mut report = analyzer::default_composite().run(&scan_data);
            let cfg = analyzer::AnalyzerConfig::from_env();
            analyzer::apply_suppressions(&mut report, &cfg, &root);
            if let Some(min) = cfg.min_severity.clone() { analyzer::filter_min_severity(&mut report, min); }
            let plan = autofix::plan_autofixes(&report.findings);
            println!("Planned actions: {}", plan.actions.len());
            for (i,a) in plan.actions.iter().enumerate() { match a { autofix::AutoFixAction::ReplaceInFile { path, .. } => println!("{}. Replace in {}", i+1, path), autofix::AutoFixAction::WriteFile { path, .. } => println!("{}. Write {}", i+1, path) }; }
            if let Some(fp) = plan_json { std::fs::write(fp, serde_json::to_string_pretty(&autofix::serialize_plan(&plan))?)?; }
            if let Some(diffp) = diff_out { let diff_text = autofix::plan_unified_diff(&root, &plan)?; std::fs::write(diffp, diff_text)?; }
            if apply { autofix::apply_plan(&root, &plan, dry_run)?; println!("Applied (dry_run={})", dry_run); }
    },
        #[cfg(feature = "semantic")]
        Commands::SemanticBuild { content_bytes, force } => {
            let si = semantic::build_semantic(&root, content_bytes, force)?; println!("Semantic index built at {}", si.dir.display());
    },
        #[cfg(feature = "semantic")]
        Commands::SemanticSearch { query, limit } => {
            let hits = semantic::semantic_search(&root, &query, limit)?; for (p, score, snip) in hits { println!("{score:.2} {p} :: {snip}"); }
    },
    Commands::Watch { content_bytes } => {
            watch::run_watch(root, content_bytes)?;
    },
        #[cfg(feature = "server")]
        Commands::Serve { port } => {
            let port = port.or_else(|| std::env::var("SERVICE_PORT").ok().and_then(|v| v.parse().ok())).unwrap_or(8080);
            server::run_server(root, port)?;
    },
        #[cfg(feature = "discord")]
        Commands::Bot {} => {
            discord_bot::run_blocking(root)?;
    },
        Commands::Prune { keep, max_age_days, dry_run } => {
            if keep == 0 && max_age_days == 0 { println!("Nothing to do (specify --keep and/or --max-age-days)"); }
            let mut snaps = persist::list_snapshots(&root)?; // sorted ascending
            let mut to_remove: Vec<std::path::PathBuf> = Vec::new();
            if keep > 0 && snaps.len() > keep {
                let count_remove = snaps.len() - keep;
                to_remove.extend_from_slice(&snaps[..count_remove]);
                snaps.drain(..count_remove); // simulate removal for next phase
            }
            if max_age_days > 0 {
                let now = chrono::Utc::now();
                for p in &snaps { if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if let Some(ts_part) = name.strip_prefix("scan-").and_then(|rest| rest.split('.').next()) {
                        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts_part, "%Y%m%dT%H%M%SZ") {
                            let dt_utc = chrono::DateTime::from_naive_utc_and_offset(dt, chrono::Utc);
                            if (now - dt_utc).num_days() > max_age_days { to_remove.push(p.clone()); }
                        }
                    }
                }}
            }
            if dry_run {
                println!("Dry run: would remove {} snapshots", to_remove.len());
                for p in &to_remove { println!(" - {}", p.file_name().and_then(|s| s.to_str()).unwrap_or("?")); }
            } else {
                if keep > 0 { persist::prune_by_count(&root, keep)?; println!("Applied count pruning -> keep {}", keep); }
                if max_age_days > 0 { persist::prune_by_age(&root, max_age_days)?; println!("Applied age pruning -> max {} days", max_age_days); }
                let remaining = persist::list_snapshots(&root)?; println!("Remaining snapshots: {}", remaining.len());
            }
    },
    Commands::Stats { format, limit, warn_ratio_increase } => {
            let mut metas = persist::load_all_snapshot_meta(&root);
            let total_count = metas.len();
            if limit>0 && metas.len() > limit { metas = metas[metas.len()-limit..].to_vec(); }
            if metas.is_empty() { println!("No snapshot metadata found."); }
            else {
                let total_orig: usize = metas.iter().map(|m| m.original_bytes).sum();
                let total_stored: usize = metas.iter().map(|m| m.stored_bytes).sum();
                let avg_ratio = if total_orig>0 { total_stored as f64 / total_orig as f64 } else { 0.0 };
                let mut ratio_increase_pct: Option<f64> = None;
                let mut ratio_warning = false;
                if metas.len() >= 2 {
                    let latest = metas.last().unwrap().compression_ratio;
                    let prev_avg = {
                        let prev = &metas[..metas.len()-1];
                        let sum: f64 = prev.iter().map(|m| m.compression_ratio).sum();
                        sum / prev.len() as f64
                    };
                    if prev_avg > 0.0 {
                        let inc = (latest - prev_avg) / prev_avg * 100.0;
                        ratio_increase_pct = Some(inc);
                        if let Some(th) = warn_ratio_increase { if inc > th { ratio_warning = true; } }
                    }
                }
                match format.as_str() {
                    "json" => {
                        let json = serde_json::json!({
                            "count": total_count,
                            "shown": metas.len(),
                            "total_original_bytes": total_orig,
                            "total_stored_bytes": total_stored,
                            "overall_compression_ratio": avg_ratio,
                            "compression_ratio_increase_pct": ratio_increase_pct,
                            "compression_ratio_warning": ratio_warning,
                            "snapshots": metas,
                        });
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("Snapshots (total={} shown={}):", total_count, metas.len());
                        println!("Total original bytes: {}", total_orig);
                        println!("Total stored bytes: {}", total_stored);
                        println!("Overall compression ratio: {:.2}%", avg_ratio*100.0);
                        if let Some(pct) = ratio_increase_pct { println!("Latest ratio delta vs prior avg: {pct:.2}%{}", if ratio_warning { "  **WARNING**" } else { "" }); }
                        println!("Recent compression trend (newest last):");
                        for m in &metas { println!(" - {} ratio={:.2}% stored={} orig={}", m.timestamp, m.compression_ratio*100.0, m.stored_bytes, m.original_bytes); }
                    }
                }
            }
    },
    Commands::Prompts { category, name, format } => {
            let cat = category.map(|s| s.to_ascii_lowercase());
            let nm = name.map(|s| s.to_ascii_lowercase());
            let cat_list = fks_analyze::prompts::list_categories();
            let mut items: Vec<&fks_analyze::prompts::PromptTemplate> = fks_analyze::prompts::catalog().prompts.iter().collect();
            if let Some(c) = &cat { items.retain(|p| p.category.to_ascii_lowercase() == *c); }
            if let Some(n) = &nm { items.retain(|p| p.name.to_ascii_lowercase().contains(n)); }
            match format.as_str() {
                "json" => {
                    let json = serde_json::json!({
                        "guidelines": fks_analyze::prompts::catalog().guidelines,
                        "categories": cat_list,
                        "count": items.len(),
                        "prompts": items.iter().map(|p| serde_json::json!({
                            "category": p.category,
                            "name": p.name,
                            "description": p.description,
                            "template": p.template,
                        })).collect::<Vec<_>>()
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                "md" | "markdown" => {
                    println!("{}", fks_analyze::prompts::to_markdown());
                }
                _ => {
                    println!("Prompt Guidelines: {}", fks_analyze::prompts::catalog().guidelines);
                    println!("Categories: {}", cat_list.join(", "));
                    for p in items { println!("\n[{}] {} - {}\n{}\n---", p.category, p.name, p.description, p.template); }
                }
            }
    },
    Commands::Prompt { name, fill, format } => {
            use fks_analyze::prompts::{fill_template, build_fill_context};
        use std::time::Duration;
            let catalog = fks_analyze::prompts::catalog();
            if let Some(p) = fks_analyze::prompts::best_match(&name) {
                // Build context if fill requested
                let mut ctx_total_files = 0usize; let mut ctx_primary_lang = String::new();
                let (filled_opt, top_exts_ctx, siblings) = if fill {
                    // Use cached builder (TTL 60s) to avoid frequent full scans
                    let ctx = build_fill_context(&root, Duration::from_secs(60));
                    ctx_total_files = ctx.total_files; ctx_primary_lang = ctx.primary_lang.clone();
                    let siblings = ctx.sibling_repos.clone();
                    let top_exts = ctx.top_exts.clone();
                    (Some(fill_template(&p.template, &ctx)), top_exts, siblings)
                } else { (None, vec![], vec![]) };
                if format == "json" {
                    let json = serde_json::json!({
                        "name": p.name,
                        "category": p.category,
                        "description": p.description,
                        "template": p.template,
                        "filled": filled_opt,
                        "top_exts": top_exts_ctx,
                        "siblings": siblings,
                        "total_files": if filled_opt.is_some() { ctx_total_files } else { 0 },
                        "primary_lang": if filled_opt.is_some() { ctx_primary_lang } else { String::new() },
                        "guidelines": catalog.guidelines,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                } else {
                    if let Some(filled) = filled_opt { println!("{}\nCategory: {}\n{}\n---\n{}", p.name, p.category, p.description, filled); }
                    else { println!("{}\nCategory: {}\n{}\n---\n{}", p.name, p.category, p.description, p.template); }
                }
            } else { eprintln!("Prompt not found: {name}"); std::process::exit(1); }
    },
    }
    Ok(())
}
