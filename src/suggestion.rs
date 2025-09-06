use crate::scan::ScanSummary;
use crate::diff::DiffResult;
use crate::analyzer::{FindingsReport, findings_to_tasks};
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Suggestions {
    pub tasks: Vec<String>,
    pub prompt_text: String,
}

pub fn generate(root: &str, scan: &ScanSummary, diff: Option<&DiffResult>, findings: Option<&FindingsReport>) -> Suggestions {
    let mut tasks: Vec<String> = Vec::new();
    if scan.total_files == 0 { tasks.push("Populate the repository with initial source files for fks_analyze (currently empty)".into()); }
    let has_readme = scan.files.iter().any(|f| f.path.eq_ignore_ascii_case("README.md"));
    if !has_readme { tasks.push("Create a README.md documenting fks_analyze purpose, CLI usage, bot commands, and architecture".into()); }
    // Heuristic: look for large counts of certain extensions & note missing tests
    let mut ext_map: BTreeMap<String, usize> = BTreeMap::new();
    for (ext,cnt) in &scan.counts_by_ext { ext_map.insert(ext.clone(), *cnt); }
    if ext_map.get("rs").copied().unwrap_or(0) > 0 { 
        let has_tests = scan.files.iter().any(|f| f.path.starts_with("tests/") || f.path.contains("_test.rs"));
        if !has_tests { tasks.push("Add Rust tests (unit + integration) for scanning and suggestion modules".into()); }
    }
    if let Some(d) = diff {
        if d.summary.added > 0 { tasks.push(format!("Review {} newly added files; ensure tests & docs updated", d.summary.added)); }
        if d.summary.modified > 0 { tasks.push(format!("Refactor or document {} modified files (consider stability)", d.summary.modified)); }
    }
    if let Some(report) = findings {
        let mut ftasks = findings_to_tasks(&report.findings);
        tasks.append(&mut ftasks);
    }
    // Core roadmap suggestions (dedup simple)
    let mut roadmap = vec![
        "Add Discord feature integration (serenity) with commands: /scan, /analyze, /suggest, /file".to_string(),
        "Implement LLM adapter (local Ollama) with a trait so suggestions can be enriched".to_string(),
        "Persist previous analyses in a lightweight SQLite or JSON log to track deltas".to_string(),
        "Add Axum HTTP endpoints: GET /health, POST /analyze, GET /suggest".to_string(),
    ];
    for r in roadmap.drain(..) { if !tasks.iter().any(|t| t.contains(&r)) { tasks.push(r); } }
    let prompt_text = format!("You are assisting on the fks_analyze Rust service. Root: {root}. Next prioritized tasks:\n1. {}\n2. {}\n3. {}\n4. {}\n5. {}\nProvide concrete code changes for the top item.",
        tasks.get(0).unwrap_or(&"(none)".into()),
        tasks.get(1).unwrap_or(&"(none)".into()),
        tasks.get(2).unwrap_or(&"(none)".into()),
        tasks.get(3).unwrap_or(&"(none)".into()),
        tasks.get(4).unwrap_or(&"(none)".into()),
    );
    Suggestions { tasks, prompt_text }
}
