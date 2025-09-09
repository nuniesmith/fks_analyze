//! High-level analysis pipeline (replacing external shell script).
use anyhow::Result;
use crate::{scan, analyzer, suggestion, diff, findings_persist};

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub root: String,
    pub content_bytes: usize,
    pub save_findings: bool,
    pub save_scan: bool,
}

#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub findings_saved: Option<String>,
    pub snapshot_saved: Option<String>,
    pub tasks: Vec<String>,
    pub counts: std::collections::BTreeMap<crate::analyzer::Severity, usize>,
}

pub fn run_pipeline(opts: PipelineOptions) -> Result<PipelineOutput> {
    // 1. Scan
    let scan_sum = scan::scan_repo(&opts.root, opts.content_bytes)?;
    let snapshot_saved = if opts.save_scan { crate::persist::save_snapshot(&opts.root, &scan_sum).ok().map(|p| p.display().to_string()) } else { None };
    // 2. Run analyzers
    let mut report = analyzer::default_composite().run(&scan_sum);
    let cfg = analyzer::AnalyzerConfig::from_env();
    analyzer::apply_suppressions(&mut report, &cfg, &opts.root);
    // Apply optional min severity filter via env config
    if let Some(ms) = cfg.min_severity.clone() { crate::analyzer::filter_min_severity(&mut report, ms); }
    let findings_saved = if opts.save_findings { findings_persist::save_findings(&opts.root, &report).ok().map(|p| p.display().to_string()) } else { None };
    // 3. Diff (optional enrichment)
    let diff_opt = diff::latest_diff(&opts.root).ok();
    // 4. Suggestions using findings
    let suggestions = suggestion::generate(&opts.root, &scan_sum, diff_opt.as_ref(), Some(&report));
    Ok(PipelineOutput { findings_saved, snapshot_saved, tasks: suggestions.tasks, counts: report.counts })
}
