use crate::analyzer::FindingsReport;
use anyhow::{Result, Context};
use chrono::Utc;
use std::{path::PathBuf, fs};

fn findings_dir(root: &str) -> PathBuf { PathBuf::from(root).join(".fks_analyze/findings") }

pub fn save_findings(root: &str, report: &FindingsReport) -> Result<PathBuf> {
    let dir = findings_dir(root); fs::create_dir_all(&dir)?;
    let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
    let path = dir.join(format!("findings-{ts}.json"));
    fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

pub fn list_findings(root: &str) -> Result<Vec<PathBuf>> {
    let dir = findings_dir(root); if !dir.exists() { return Ok(vec![]); }
    let mut items: Vec<_> = fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json")).collect();
    items.sort();
    Ok(items)
}

pub fn load_findings(path: &PathBuf) -> Result<FindingsReport> {
    let data = fs::read(path).with_context(|| format!("reading findings snapshot {:?}", path))?;
    Ok(serde_json::from_slice(&data)?)
}