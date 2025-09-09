use serde::{Serialize, Deserialize};
use std::path::Path;
use walkdir::WalkDir;
use rayon::prelude::*;
use globset::{GlobBuilder, GlobSetBuilder};
use anyhow::Result;
use std::fs;
use std::io::Read;
// (mtime currently unused)

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub ext: Option<String>,
    #[serde(skip_serializing_if="Option::is_none")] pub snippet: Option<String>,
    #[serde(skip_serializing_if="Option::is_none")] pub hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ScanSummary {
    pub root: String,
    pub files: Vec<FileEntry>,
    pub counts_by_ext: Vec<(String, usize)>,
    pub total_files: usize,
    pub total_size: u64,
    #[serde(default)] pub cache_reused: usize,
    #[serde(default)] pub duration_ms: u128,
}

pub fn scan_repo(root: &str, content_bytes: usize) -> Result<ScanSummary> {
    scan_repo_opts(root, content_bytes, false)
}

pub fn scan_repo_opts(root: &str, content_bytes: usize, skip_shared: bool) -> Result<ScanSummary> {
    scan_repo_with_ignore(root, content_bytes, skip_shared, &[])
}

pub fn scan_repo_with_ignore(root: &str, content_bytes: usize, skip_shared: bool, ignore_globs: &[String]) -> Result<ScanSummary> {
    let mut summary = ScanSummary { root: root.to_string(), ..Default::default() };
    let started = std::time::Instant::now();
    let cache_dir = Path::new(root).join(".fks_analyze/cache");
    let cache_file = cache_dir.join("scan_cache.json");
    let mut previous: Option<ScanSummary> = None;
    if cache_file.exists() { if let Ok(data)=fs::read(&cache_file) { if let Ok(old)=serde_json::from_slice::<ScanSummary>(&data) { previous=Some(old); } } }
    let mut reused = 0usize; // number of reused cached entries
    let hash_max_bytes: u64 = std::env::var("FKS_HASH_MAX_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(5_000_000);
    let ignore = build_ignore(ignore_globs)?;
    let par_enabled = std::env::var("FKS_SCAN_PAR").ok().as_deref() == Some("1");
    let walker: Vec<_> = WalkDir::new(root).follow_links(false).into_iter().filter_map(|e| e.ok()).collect();
    let full_capture = std::env::var("FKS_SNAPSHOT_FULL").ok().as_deref() == Some("1");
    let selective_full = std::env::var("FKS_SNAPSHOT_FULL_SELECTIVE").ok().as_deref() == Some("1");
    let full_capture_max: usize = std::env::var("FKS_SNAPSHOT_FULL_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let process_entry = |entry: &walkdir::DirEntry| -> Option<(FileEntry, bool)> {
        let p = entry.path();
        if !p.is_file() { return None; }
        if skip_path(p) { return None; }
        if skip_shared && is_shared_subdir(root, p) { return None; }
        if is_ignored(p, root, &ignore) { return None; }
        let meta = entry.metadata().ok()?; let size = meta.len();
        let rel_path = p.strip_prefix(root).unwrap_or(p).to_string_lossy().to_string();
        let hash = if size <= hash_max_bytes {
            let mut file = fs::File::open(p).ok();
            let mut hasher = blake3::Hasher::new();
            if let Some(ref mut fh) = file {
                let mut buf = [0u8; 8192];
                while let Ok(n) = fh.read(&mut buf) { if n==0 { break; } hasher.update(&buf[..n]); }
                Some(hasher.finalize().to_hex().to_string())
            } else { None }
        } else { None };
        if let Some(prev) = &previous { if let Some(old) = prev.files.iter().find(|f| f.path == rel_path) {
            let same = if old.hash.is_some() && hash.is_some() { old.hash == hash } else { old.size == size && old.hash.is_none() && hash.is_none() };
            if same { return Some((old.clone(), true)); }
        }}
        let ext = p.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase());
    // Decide capture policy:
    // 1. If selective_full enabled: capture full only if changed (hash differs or no previous record)
    // 2. Else if full_capture: capture full
    // 3. Else capture limited snippet via content_bytes
    let changed = previous.as_ref().and_then(|prev| prev.files.iter().find(|f| f.path==rel_path)).map(|old| {
        if let (Some(h1), Some(h2)) = (&old.hash, &hash) { h1!=h2 } else { old.size != size }
    }).unwrap_or(true);
    let want_full = if selective_full { changed && (size as usize) <= full_capture_max } else { full_capture && (size as usize) <= full_capture_max };
    let snippet = if (content_bytes>0 && size>0 && size as usize <= content_bytes && !want_full) || want_full {
            // Read raw bytes then truncate safely at UTF-8 boundary to avoid panics
            fs::read(p).ok().and_then(|mut bytes| {
        let limit = if want_full { full_capture_max } else { content_bytes };
        if bytes.len() > limit { bytes.truncate(limit); }
                match String::from_utf8(bytes) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        // Take valid prefix
                        let valid = e.into_bytes();
                        String::from_utf8(valid).ok()
                    }
                }
            })
        } else { None };
        Some((FileEntry { path: rel_path, size, ext, snippet, hash }, false))
    };
    let collected: Vec<(FileEntry,bool)> = if par_enabled {
        walker.par_iter().filter_map(process_entry).collect()
    } else { walker.iter().filter_map(process_entry).collect() };
    for (fe, was_reused) in collected { if was_reused { reused +=1; } summary.files.push(fe); }
    summary.total_files = summary.files.len();
    summary.total_size = summary.files.iter().map(|f| f.size).sum();
    use std::collections::HashMap; let mut map: HashMap<String, usize> = HashMap::new();
    for f in &summary.files { map.entry(f.ext.clone().unwrap_or_else(||"_noext".into())).or_default().add_assign(1);}    
    let mut counts: Vec<_> = map.into_iter().collect();
    counts.sort_by(|a,b| b.1.cmp(&a.1));
    summary.counts_by_ext = counts;
    // Persist cache (ignore snippet content_bytes > 4096 for size control)
    if !cache_dir.exists() { let _=fs::create_dir_all(&cache_dir); }
    summary.cache_reused = reused;
    summary.duration_ms = started.elapsed().as_millis();
    if let Ok(bytes) = serde_json::to_vec(&summary) { let _ = fs::write(&cache_file, bytes); }
    Ok(summary)
}

fn skip_path(p: &Path) -> bool {
    let s = p.to_string_lossy();
    const SKIP: &[&str] = &["/.git/","/target/","/node_modules/","/venv/","/.mypy_cache/","/.pytest_cache/","/dist/","/build/"]; 
    SKIP.iter().any(|k| s.contains(k))
}

fn build_ignore(patterns: &[String]) -> Result<globset::GlobSet> {
    let mut b = GlobSetBuilder::new();
    for pat in patterns { if !pat.trim().is_empty() { b.add(GlobBuilder::new(pat).case_insensitive(true).build()?); }}
    Ok(b.build()?)
}

fn is_ignored(p: &Path, root: &str, set: &globset::GlobSet) -> bool {
    if set.is_empty() { return false; }
    if let Ok(rel) = p.strip_prefix(root) { if let Some(s)=rel.to_str() { return set.is_match(s); }}
    false
}

fn is_shared_subdir(root: &str, p: &Path) -> bool {
    if let Ok(rel) = p.strip_prefix(root) { 
        let comps: Vec<_> = rel.components().collect();
        if comps.first().and_then(|c| c.as_os_str().to_str()) == Some("shared") { return true; }
    }
    false
}

// Provide AddAssign for usize locally (nightly has stable) - implement manually
trait AddAssignExt { fn add_assign(&mut self, v: usize); }
impl AddAssignExt for usize { fn add_assign(&mut self, v: usize) { *self += v; }}

