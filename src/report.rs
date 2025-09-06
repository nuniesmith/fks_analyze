use anyhow::Result;
use crate::{analyzer, findings_persist, scan, diff};
use chrono::Utc;

pub struct ReportOptions { pub root: String, pub limit: usize, pub trend: usize, pub content_bytes: usize, pub min_severity: Option<analyzer::Severity>, pub kinds: Vec<String>, pub format: ReportFormat, pub show_diff: bool }

pub enum ReportFormat { Markdown, Json, Html, Csv }

pub fn generate_report(opts: ReportOptions) -> Result<String> {
    // Load or build findings
    let paths = findings_persist::list_findings(&opts.root)?;
    let (mut report, generated) = if let Some(p) = paths.last() {
        (findings_persist::load_findings(p)?, false)
    } else {
        let sc = scan::scan_repo(&opts.root, opts.content_bytes)?;
        let mut rep = analyzer::default_composite().run(&sc);
        let cfg = analyzer::AnalyzerConfig::from_env();
        analyzer::apply_suppressions(&mut rep, &cfg, &opts.root);
        (rep, true)
    };
    // Apply optional severity/kind filters (after suppression)
    if let Some(ms) = opts.min_severity.clone() { analyzer::filter_min_severity(&mut report, ms); }
    if !opts.kinds.is_empty() {
        let before = report.findings.len();
        let allowed: Vec<String> = opts.kinds.iter().map(|s| s.to_ascii_lowercase()).collect();
        report.findings.retain(|f| f.kind.as_ref().map(|k| {
            let kstr = format!("{:?}", k).to_ascii_lowercase();
            allowed.contains(&kstr)
        }).unwrap_or(false));
        if before != report.findings.len() { report.meta.insert("filtered_by_kind".into(), (before - report.findings.len()).to_string()); }
        // recompute counts
        let mut counts: std::collections::BTreeMap<analyzer::Severity, usize> = std::collections::BTreeMap::new();
        for f in &report.findings { *counts.entry(f.severity.clone()).or_insert(0)+=1; }
        report.counts = counts;
    }
    // Focus tasks
    let focus = analyzer::top_focus_tasks(&report, opts.limit);
    // Trend rows
    let trend_paths = if paths.len() > opts.trend { &paths[paths.len()-opts.trend..] } else { &paths[..] };
    let mut trend_rows: Vec<(String, usize, usize, usize, usize, usize)> = Vec::new();
    for p in trend_paths { if let Ok(rep) = findings_persist::load_findings(p) { use analyzer::Severity; let fname=p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string(); let get=|s:&Severity| rep.counts.get(s).copied().unwrap_or(0); trend_rows.push((fname,get(&Severity::Info),get(&Severity::Low),get(&Severity::Medium),get(&Severity::High),get(&Severity::Critical))); }}
    // Attempt latest diff (without unified hunks by default). We'll try to enrich if snippets exist.
    let (latest_diff, latest_diff_unified) = if opts.show_diff {
        let base = diff::compute_from_args(&opts.root, None, None).ok();
        let uni = if let Ok((d, from_scan, to_scan)) = diff::compute_with_scans_from_args(&opts.root, None, None) { Some(diff::enrich_with_unified_snippets(d, &from_scan, &to_scan, 50_000, 3)) } else { None };
        (base, uni)
    } else { (None, None) };

    // Findings severity delta (compare last two findings snapshots if present)
    let mut severity_delta_json = None;
    let mut severity_delta_md = String::new();
    if let Ok(flist) = findings_persist::list_findings(&opts.root) {
        if flist.len() >= 2 {
            if let (Ok(prev), Ok(curr)) = (findings_persist::load_findings(&flist[flist.len()-2]), findings_persist::load_findings(&flist[flist.len()-1])) {
                use analyzer::Severity; let sevs = [Severity::Info, Severity::Low, Severity::Medium, Severity::High, Severity::Critical];
                let mut deltas = Vec::new();
                for s in sevs.iter() {
                    let a = prev.counts.get(s).copied().unwrap_or(0) as i64; let b = curr.counts.get(s).copied().unwrap_or(0) as i64; deltas.push((format!("{:?}", s), a, b, b-a));
                }
                severity_delta_json = Some(deltas.iter().map(|(n,a,b,d)| serde_json::json!({"severity":n,"previous":a,"current":b,"delta":d})).collect::<Vec<_>>());
                use std::fmt::Write; write!(severity_delta_md, "\n## Severity Delta (prev -> current)\n\n| Severity | Prev | Curr | Δ |\n|---|---:|---:|---:|\n").ok(); for (n,a,b,d) in deltas { write!(severity_delta_md, "| {} | {} | {} | {} |\n", n, a, b, d).ok(); }
            }
        }
    }

    if let ReportFormat::Json = opts.format {
        // Attach latest snapshot meta if available
        let latest_meta = crate::persist::list_snapshots(&opts.root).ok().and_then(|l| l.last().cloned()).and_then(|p| crate::persist::load_snapshot_meta_for(&p));
        let json = serde_json::json!({
            "generated_at": report.meta.get("generated_at"),
            "root": opts.root,
            "counts": report.counts,
            "meta": report.meta,
            "findings": report.findings,
            "focus_tasks": focus,
            "trend": trend_rows.iter().map(|r| serde_json::json!({"snapshot": r.0, "info": r.1, "low": r.2, "medium": r.3, "high": r.4, "critical": r.5})).collect::<Vec<_>>(),
            "latest_diff": latest_diff,
            "latest_diff_unified": latest_diff_unified,
            "severity_delta": severity_delta_json,
            "latest_snapshot_meta": latest_meta,
        });
        return Ok(serde_json::to_string_pretty(&json)?);
    }
    if let ReportFormat::Html = opts.format {
        use std::fmt::Write;
        let mut html = String::new();
    writeln!(html, "<!DOCTYPE html><html><head><meta charset='utf-8'><title>FKS Analyze Report</title><style>body{{font-family:Arial,sans-serif;margin:1.5rem;}} table{{border-collapse:collapse;margin:1rem 0;}} th,td{{border:1px solid #ccc;padding:4px 8px;font-size:0.9rem;}} th{{background:#f5f5f5;}} .sev-Info{{color:#666}} .sev-Low{{color:#1b6}} .sev-Medium{{color:#d80}} .sev-High{{color:#d40}} .sev-Critical{{color:#b00;font-weight:bold}} code{{background:#f0f0f0;padding:2px 4px;border-radius:4px;}} .delta-pos{{color:#b00;font-weight:bold;}} .delta-neg{{color:#080;font-weight:bold;}} .delta-zero{{color:#555;}} caption{{font-weight:bold;margin-bottom:4px;}} details summary{{cursor:pointer;font-weight:bold;}} .small{{font-size:0.75rem;color:#666;margin-top:2rem;}}</style></head><body>")?;
        writeln!(html, "<h1>FKS Analyze Dashboard</h1><p><strong>Root:</strong> <code>{}</code></p>", opts.root)?;
    writeln!(html, "<h2>Findings Summary</h2><ul>")?; for (sev,count) in &report.counts { writeln!(html, "<li class='sev-{0:?}'><strong>{0:?}</strong>: {1}</li>", sev, count)?; } writeln!(html, "</ul>")?;
        writeln!(html, "<h2>Top Findings</h2><ul>")?; for f in report.findings.iter().take(20) { writeln!(html, "<li class='sev-{0:?}'><strong>[{0:?}] {1}</strong> {2}</li>", f.severity, f.title, f.path.as_deref().unwrap_or(""))?; } writeln!(html, "</ul>")?;
        if !trend_rows.is_empty() { writeln!(html, "<h2>Trend (last {} snapshots)</h2><table><tr><th>Snapshot</th><th>Info</th><th>Low</th><th>Med</th><th>High</th><th>Crit</th></tr>", trend_rows.len())?; for r in &trend_rows { writeln!(html, "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", r.0,r.1,r.2,r.3,r.4,r.5)?; } writeln!(html, "</table>")?; }
        if !focus.is_empty() { writeln!(html, "<h2>Focus Tasks</h2><ol>")?; for t in &focus { writeln!(html, "<li>{}</li>", t)?; } writeln!(html, "</ol>")?; }
        if let Some(d) = &latest_diff_unified.or(latest_diff) { writeln!(html, "<h2>Latest Diff</h2><p>+{} -{} ~{} bytes Δ={}</p>", d.summary.added, d.summary.removed, d.summary.modified, d.summary.bytes_delta)?; }
        // Compression trend
        let metas = crate::persist::load_all_snapshot_meta(&opts.root); if !metas.is_empty() { writeln!(html, "<h2>Compression Trend</h2><table><tr><th>Timestamp</th><th>Compressed</th><th>Orig KB</th><th>Stored KB</th><th>Ratio %</th></tr>")?; for m in metas.iter().rev().take(10).rev() { writeln!(html, "<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{:.2}</td></tr>", m.timestamp, m.is_compressed, m.original_bytes as f64/1024.0, m.stored_bytes as f64/1024.0, m.compression_ratio*100.0)?; } writeln!(html, "</table>")?; }
        if let Some(sd) = &severity_delta_json { if !sd.is_empty() {
            writeln!(html, "<h2>Severity Delta</h2><table><tr><th>Severity</th><th>Prev</th><th>Curr</th><th>Δ</th></tr>")?;
            for row in sd { if let (Some(sev), Some(prev), Some(curr), Some(delta)) = (row.get("severity"), row.get("previous"), row.get("current"), row.get("delta"),) {
                let d = delta.as_i64().unwrap_or(0); let cls = if d>0 {"delta-pos"} else if d<0 {"delta-neg"} else {"delta-zero"};
                writeln!(html, "<tr><td class='sev-{0}'>{0}</td><td>{1}</td><td>{2}</td><td class='{3}'>{4}</td></tr>", sev.as_str().unwrap_or("?"), prev, curr, cls, d)?;
            }}
            writeln!(html, "</table>")?; }
        }
        writeln!(html, "<p class='small'><em>Generated at {}</em></p></body></html>", Utc::now().to_rfc3339())?;
        return Ok(html);
    }
    if let ReportFormat::Csv = opts.format {
        // CSV: sections separated by blank lines. Provide counts, findings (limited), trend, severity delta.
        let mut csv = String::new();
        use std::fmt::Write;
        // Counts
        writeln!(csv, "section,type,count").ok();
        for (sev,count) in &report.counts { writeln!(csv, "counts,{:?},{}", sev, count).ok(); }
        // Findings (limit 100 for CSV)
        writeln!(csv, "\nsection,severity,title,path,kind,score").ok();
        for f in report.findings.iter().take(100) {
            let path = f.path.as_deref().unwrap_or("").replace(',', ";");
            writeln!(csv, "finding,{:?},{} ,{} ,{} ,{}", f.severity, f.title.replace(',', ";"), path, f.kind.as_ref().map(|k| format!("{:?}", k)).unwrap_or_default(), f.score.unwrap_or(0.0)).ok();
        }
        // Trend
        if !trend_rows.is_empty() {
            writeln!(csv, "\nsection,snapshot,info,low,medium,high,critical").ok();
            for r in &trend_rows { writeln!(csv, "trend,{},{},{},{},{},{}", r.0,r.1,r.2,r.3,r.4,r.5).ok(); }
        }
        // Severity delta
        if let Some(sd) = &severity_delta_json { if !sd.is_empty() { writeln!(csv, "\nsection,severity,previous,current,delta").ok(); for row in sd { writeln!(csv, "severity_delta,{},{},{},{}", row.get("severity").and_then(|v| v.as_str()).unwrap_or("?"), row.get("previous").and_then(|v| v.as_i64()).unwrap_or(0), row.get("current").and_then(|v| v.as_i64()).unwrap_or(0), row.get("delta").and_then(|v| v.as_i64()).unwrap_or(0)).ok(); }} }
        return Ok(csv);
    }
    // Build markdown
    use std::fmt::Write;
    let mut md = String::new();
    writeln!(md, "# FKS Analyze Dashboard\n")?;
    writeln!(md, "Generated at: {}  ", Utc::now().to_rfc3339())?;
    writeln!(md, "Root: `{}`  ", opts.root)?;
    if generated { writeln!(md, "(Findings generated fresh this run.)\n")?; }
    // Summary counts
    writeln!(md, "## Findings Summary\n")?;
    for (sev, count) in &report.counts { writeln!(md, "- {:?}: {}", sev, count)?; }
    if let Some(s) = report.meta.get("suppressed") { writeln!(md, "- Suppressed (this run): {}", s)?; }
    // Top findings details (limit 20 for brevity)
    writeln!(md, "\n## Top Findings (score ordered)\n")?;
    for f in report.findings.iter().take(20) { writeln!(md, "- {:.1} [{:?}] **{}** {}", f.score.unwrap_or(0.0), f.severity, f.title, f.path.as_deref().unwrap_or(""))?; }
    writeln!(md, "\n## Focus Tasks (top {})\n", focus.len())?;
    for (i,t) in focus.iter().enumerate() { writeln!(md, "{}. {}", i+1, t)?; }
    // Trend table
    if !trend_rows.is_empty() { writeln!(md, "\n## Trend (last {} snapshots)\n", trend_rows.len())?; writeln!(md, "| Snapshot | Info | Low | Med | High | Crit |\n|---|---:|---:|---:|---:|---:|")?; for r in trend_rows { writeln!(md, "| {} | {} | {} | {} | {} | {} |", r.0, r.1, r.2, r.3, r.4, r.5)?; } }
    // Latest scan stats (if snapshot exists)
    if let Some(scan_path) = crate::persist::list_snapshots(&opts.root)?.last().cloned() {
        if let Ok(scan) = crate::persist::load_snapshot(&scan_path) {
            writeln!(md, "\n## Latest Scan Stats\n")?;
            writeln!(md, "- snapshot: {}", scan_path.file_name().and_then(|s| s.to_str()).unwrap_or("?"))?;
            writeln!(md, "- total_files: {}", scan.total_files)?;
            writeln!(md, "- total_size_bytes: {}", scan.total_size)?;
            writeln!(md, "- cache_reused: {}", scan.cache_reused)?;
            writeln!(md, "- duration_ms: {}", scan.duration_ms)?;
            if let Some(meta) = crate::persist::load_snapshot_meta_for(&scan_path) {
                writeln!(md, "- compressed: {}", meta.is_compressed)?;
                writeln!(md, "- original_bytes: {}", meta.original_bytes)?;
                writeln!(md, "- stored_bytes: {}", meta.stored_bytes)?;
                writeln!(md, "- compression_ratio: {:.2}%", meta.compression_ratio*100.0)?;
            }
        }
    }
    // Diff summary
    if opts.show_diff { if let Some(d) = &latest_diff_unified.or(latest_diff) {
        writeln!(md, "\n## Latest Diff ({}) -> ({})\n", d.from, d.to)?;
        writeln!(md, "Files: +{} -{} ~{} bytes_delta={}  ", d.summary.added, d.summary.removed, d.summary.modified, d.summary.bytes_delta)?;
        if let Some(u) = &d.unified { for f in u.iter().take(5) { writeln!(md, "\n### Diff: `{}`\n", f.path)?; for h in &f.hunks { writeln!(md, "````diff")?; writeln!(md, "{}", h.header)?; for l in &h.lines { writeln!(md, "{}", l)?; } writeln!(md, "````\n")?; } } }
    }}
    // Compression trend (last 10 metas)
    let metas = crate::persist::load_all_snapshot_meta(&opts.root);
    if !metas.is_empty() {
        let take = metas.iter().rev().take(10).cloned().collect::<Vec<_>>();
        writeln!(md, "\n## Compression Trend (last {} snapshots)\n", take.len())?;
        writeln!(md, "| Timestamp | Compressed | Orig (KB) | Stored (KB) | Ratio % |\n|---|:---:|---:|---:|---:|")?;
        for m in take.into_iter().rev() { writeln!(md, "| {} | {} | {:.1} | {:.1} | {:.2} |", m.timestamp, m.is_compressed, m.original_bytes as f64/1024.0, m.stored_bytes as f64/1024.0, m.compression_ratio*100.0)?; }
    }
    if !severity_delta_md.is_empty() { md.push_str(&severity_delta_md); }

    writeln!(md, "\n## Configuration\n")?;
    let cfg = analyzer::AnalyzerConfig::from_env();
    writeln!(md, "- large_file_threshold: {}", cfg.large_file_threshold)?;
    writeln!(md, "- git_churn_max_files: {}", cfg.git_churn_max_files)?;
    writeln!(md, "- git_churn_min_commits: {}", cfg.git_churn_min_commits)?;
    writeln!(md, "- suppressed_ids: {}", cfg.suppress_ids.len())?;
    if let Some(ms) = &cfg.min_severity { writeln!(md, "- min_severity: {:?}", ms)?; }
    writeln!(md, "- age_decay_start_days: {}", cfg.age_decay_start_days)?;
    writeln!(md, "- age_decay_end_days: {}", cfg.age_decay_end_days)?;
    writeln!(md, "- age_decay_min_factor: {}", cfg.age_decay_min_factor)?;
    writeln!(md, "\n---\nGenerated by `fks_analyze report`.\n")?;
    Ok(md)
}
