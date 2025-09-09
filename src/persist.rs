use crate::scan::ScanSummary;
use anyhow::{Result, Context};
use std::{path::PathBuf, fs};
use chrono::Utc;
use serde::{Serialize, Deserialize};

// Optional compression & retention controlled via env:
// FKS_SNAPSHOT_COMPRESS=1 -> store as .json.gz
// FKS_SNAPSHOT_RETENTION (integer) -> keep only latest N snapshots (delete older)
// FKS_SNAPSHOT_RETENTION_DAYS (integer) -> prune snapshots older than N days
// Both retention rules applied after saving (count first, then age)

fn history_dir(root: &str) -> PathBuf { PathBuf::from(root).join(".fks_analyze/history") }

#[derive(Debug, Default, Clone)]
pub struct SnapshotSaveOptions {
    pub compress: Option<bool>,
    pub retention: Option<usize>,
    pub retention_days: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SnapshotMeta {
    pub timestamp: String,
    pub is_compressed: bool,
    pub original_bytes: usize,
    pub stored_bytes: usize,
    pub compression_ratio: f64, // stored/original (<=1.0)
}

pub fn save_snapshot(root: &str, scan: &ScanSummary) -> Result<PathBuf> {
    save_snapshot_with(root, scan, &SnapshotSaveOptions::default())
}

pub fn save_snapshot_with(root: &str, scan: &ScanSummary, opts: &SnapshotSaveOptions) -> Result<PathBuf> {
    let dir = history_dir(root); fs::create_dir_all(&dir)?;
    let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
    let compress_env = std::env::var("FKS_SNAPSHOT_COMPRESS").ok().map(|v| v=="1" || v.to_ascii_lowercase()=="true").unwrap_or(false);
    let compress = opts.compress.unwrap_or(compress_env);
    let path = dir.join(if compress { format!("scan-{ts}.json.gz") } else { format!("scan-{ts}.json") });
    let meta_path = dir.join(format!("scan-{ts}.meta.json"));
    let mut meta = SnapshotMeta { timestamp: ts.to_string(), is_compressed: compress, ..Default::default() };
    if compress {
        use flate2::{write::GzEncoder, Compression};
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        let orig = serde_json::to_vec_pretty(scan)?; meta.original_bytes = orig.len();
        use std::io::Write; enc.write_all(&orig)?; let bytes = enc.finish()?; meta.stored_bytes = bytes.len();
        meta.compression_ratio = if meta.original_bytes>0 { meta.stored_bytes as f64 / meta.original_bytes as f64 } else { 1.0 };
        fs::write(&path, bytes)?;
        println!("snapshot saved (compressed): {} (orig={} bytes, compressed={} bytes, ratio {:.2}%)", path.display(), meta.original_bytes, meta.stored_bytes, meta.compression_ratio*100.0);
    } else {
        let data = serde_json::to_vec_pretty(scan)?; meta.original_bytes = data.len(); meta.stored_bytes = data.len(); meta.compression_ratio = 1.0; fs::write(&path, &data)?; println!("snapshot saved: {} ({} bytes)", path.display(), meta.original_bytes);
    }
    // Sidecar meta (ignore errors)
    if let Ok(j) = serde_json::to_vec_pretty(&meta) { let _ = fs::write(&meta_path, j); }
    // Determine retention preferences (CLI overrides env)
    let retention = opts.retention.or_else(|| std::env::var("FKS_SNAPSHOT_RETENTION").ok().and_then(|v| v.parse::<usize>().ok()));
    let retention_days = opts.retention_days.or_else(|| std::env::var("FKS_SNAPSHOT_RETENTION_DAYS").ok().and_then(|v| v.parse::<i64>().ok()));
    if let Some(n) = retention { prune_by_count(root, n)?; }
    if let Some(d) = retention_days { prune_by_age(root, d)?; }
    Ok(path)
}

pub fn list_snapshots(root: &str) -> Result<Vec<PathBuf>> {
    let dir = history_dir(root); if !dir.exists() { return Ok(vec![]); }
    let mut items: Vec<_> = fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| {
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".meta.json") { return false; }
            }
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) { ext=="json" || ext=="gz" } else { false }
        }).collect();
    items.sort();
    Ok(items)
}

pub fn load_snapshot(path: &PathBuf) -> Result<ScanSummary> {
    let data = fs::read(path).with_context(|| format!("reading snapshot {:?}", path))?;
    let content = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        use flate2::read::GzDecoder; use std::io::Read; let mut d = GzDecoder::new(&data[..]); let mut out = Vec::new(); d.read_to_end(&mut out)?; out
    } else { data };
    Ok(serde_json::from_slice(&content)?)
}

pub fn load_snapshot_meta_for(path: &PathBuf) -> Option<SnapshotMeta> {
    // Derive meta path: scan-TS.json or scan-TS.json.gz -> scan-TS.meta.json
    if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
        let base = if let Some(stripped) = fname.strip_suffix(".json.gz") { stripped } else if let Some(stripped) = fname.strip_suffix(".json") { stripped } else { return None; };
        let meta_name = format!("{base}.meta.json");
        let meta_path = path.parent().unwrap_or_else(|| std::path::Path::new("")).join(meta_name);
        if meta_path.exists() { if let Ok(bytes) = fs::read(meta_path) { if let Ok(meta) = serde_json::from_slice::<SnapshotMeta>(&bytes) { return Some(meta); } } }
    }
    None
}

pub fn prune_by_count(root: &str, keep: usize) -> Result<()> {
    if keep == 0 { return Ok(()); }
    let mut snaps = list_snapshots(root)?;
    if snaps.len() <= keep { return Ok(()); }
    let to_delete = snaps.drain(.. snaps.len() - keep).collect::<Vec<_>>();
    for p in to_delete { remove_snapshot_and_meta(&p); }
    Ok(())
}

pub fn prune_by_age(root: &str, max_age_days: i64) -> Result<()> {
    if max_age_days <= 0 { return Ok(()); }
    let snaps = list_snapshots(root)?; let now = Utc::now();
    for p in snaps {
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if let Some(ts_part) = name.strip_prefix("scan-") { if let Some(ts_str) = ts_part.split('.').next() {
                if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts_str, "%Y%m%dT%H%M%SZ") {
                    let dt_utc = chrono::DateTime::from_naive_utc_and_offset(dt, Utc);
                    if (now - dt_utc).num_days() > max_age_days { remove_snapshot_and_meta(&p); }
                }
            }}
        }
    }
    Ok(())
}

fn remove_snapshot_and_meta(p: &PathBuf) {
    let _ = fs::remove_file(p);
    if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
        let base = if let Some(stripped) = fname.strip_suffix(".json.gz") { stripped } else if let Some(stripped) = fname.strip_suffix(".json") { stripped } else { fname };
        let meta = p.parent().unwrap_or_else(|| std::path::Path::new(""))
            .join(format!("{base}.meta.json"));
        if meta.exists() { let _ = fs::remove_file(meta); }
    }
}

pub fn load_all_snapshot_meta(root: &str) -> Vec<SnapshotMeta> {
    if let Ok(snaps) = list_snapshots(root) {
        let mut metas: Vec<(String, SnapshotMeta)> = snaps.into_iter().filter_map(|p| load_snapshot_meta_for(&p).map(|m| (m.timestamp.clone(), m))).collect();
        metas.sort_by(|a,b| a.0.cmp(&b.0));
        metas.into_iter().map(|(_,m)| m).collect()
    } else { vec![] }
}

