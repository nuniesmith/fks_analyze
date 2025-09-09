use crate::persist;
use crate::scan::{ScanSummary, FileEntry};
use anyhow::{Result, anyhow};
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct DiffResult {
    pub from: String,
    pub to: String,
    pub added: Vec<FileEntry>,
    pub removed: Vec<FileEntry>,
    pub modified: Vec<ModifiedFile>,
    pub summary: DiffSummary,
    #[serde(skip_serializing_if="Option::is_none")] pub unified: Option<Vec<FileUnifiedDiff>>, // optional per-file diffs
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct ModifiedFile { pub path: String, pub old_size: u64, pub new_size: u64 }

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct FileUnifiedDiff { pub path: String, pub hunks: Vec<DiffHunk> }

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct DiffHunk { pub header: String, pub lines: Vec<String> }

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct DiffSummary { pub added: usize, pub removed: usize, pub modified: usize, pub bytes_delta: i64 }

pub fn compute(from: &ScanSummary, to: &ScanSummary, from_name: String, to_name: String) -> DiffResult {
    let mut map_from: HashMap<&str, &FileEntry> = HashMap::new();
    for f in &from.files { map_from.insert(&f.path, f); }
    let mut map_to: HashMap<&str, &FileEntry> = HashMap::new();
    for f in &to.files { map_to.insert(&f.path, f); }
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut modified = Vec::new();
    for (p, f) in &map_to { if !map_from.contains_key(p) { added.push((*f).clone()); } }
    for (p, f) in &map_from { if !map_to.contains_key(p) { removed.push((*f).clone()); } }
    for (p, newf) in &map_to { if let Some(oldf) = map_from.get(p) {
        let changed = if oldf.hash.is_some() && newf.hash.is_some() {
            oldf.hash != newf.hash
        } else {
            oldf.size != newf.size
        };
        if changed { modified.push(ModifiedFile { path: (*p).to_string(), old_size: oldf.size, new_size: newf.size }); }
    } }
    let bytes_delta: i64 = added.iter().map(|f| f.size as i64).sum::<i64>() - removed.iter().map(|f| f.size as i64).sum::<i64>() + modified.iter().map(|m| (m.new_size as i64 - m.old_size as i64)).sum::<i64>();
    DiffResult { 
        from: from_name, to: to_name,
        summary: DiffSummary { added: added.len(), removed: removed.len(), modified: modified.len(), bytes_delta },
        added, removed, modified,
        unified: None,
    }
}

pub fn latest_diff(root: &str) -> Result<DiffResult> {
    let snaps = persist::list_snapshots(root)?; if snaps.len() < 2 { return Err(anyhow!("Not enough snapshots")); }
    let to_path = snaps.last().unwrap().clone();
    let from_path = snaps.get(snaps.len()-2).unwrap().clone();
    let from_scan = persist::load_snapshot(&from_path)?;
    let to_scan = persist::load_snapshot(&to_path)?;
    Ok(compute(&from_scan, &to_scan, file_label(&from_path), file_label(&to_path)))
}

pub fn compute_from_args(root: &str, from: Option<String>, to: Option<String>) -> Result<DiffResult> {
    let snaps = persist::list_snapshots(root)?;
    if snaps.len() < 2 { return Err(anyhow!("Not enough snapshots for diff")); }
    let resolve = |name: Option<String>, default: PathBuf| -> PathBuf {
        if let Some(n) = name { 
            snaps.iter().find(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.contains(&n)).unwrap_or(false)).cloned().unwrap_or(default)
        } else { default }
    };
    let to_path = resolve(to, snaps.last().unwrap().clone());
    let from_path = resolve(from, snaps.get(snaps.len()-2).unwrap().clone());
    let from_scan = persist::load_snapshot(&from_path)?;
    let to_scan = persist::load_snapshot(&to_path)?;
    Ok(compute(&from_scan, &to_scan, file_label(&from_path), file_label(&to_path)))
}

fn file_label(p: &PathBuf) -> String { p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string() }

impl DiffResult {
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Diff {} -> {} | +{} -{} ~{} bytes_delta={}\n", self.from, self.to, self.summary.added, self.summary.removed, self.summary.modified, self.summary.bytes_delta));
        if !self.added.is_empty() { s.push_str("Added:\n"); for f in self.added.iter().take(10) { s.push_str(&format!("  + {} ({} bytes)\n", f.path, f.size)); } }
        if !self.removed.is_empty() { s.push_str("Removed:\n"); for f in self.removed.iter().take(10) { s.push_str(&format!("  - {}\n", f.path)); } }
        if !self.modified.is_empty() { s.push_str("Modified:\n"); for m in self.modified.iter().take(10) { s.push_str(&format!("  ~ {} ({} -> {})\n", m.path, m.old_size, m.new_size)); } }
        if let Some(unified) = &self.unified { for f in unified.iter().take(5) { s.push_str(&format!("--- a/{0}\n+++ b/{0}\n", f.path)); for h in &f.hunks { s.push_str(&format!("{}\n", h.header)); for l in &h.lines { s.push_str(l); s.push('\n'); } } } }
        s
    }
}
// NOTE: Historical legacy helper removed (now superseded by snippet-based diff logic below).

// --- Enhanced Unified Diff Using Snapshot Snippets ---

pub fn compute_with_scans_from_args(root: &str, from: Option<String>, to: Option<String>) -> Result<(DiffResult, ScanSummary, ScanSummary)> {
    let snaps = persist::list_snapshots(root)?;
    if snaps.len() < 2 { return Err(anyhow!("Not enough snapshots for diff")); }
    let resolve = |name: Option<String>, default: PathBuf| -> PathBuf {
        if let Some(n) = name {
            snaps.iter().find(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.contains(&n)).unwrap_or(false)).cloned().unwrap_or(default)
        } else { default }
    };
    let to_path = resolve(to, snaps.last().unwrap().clone());
    let from_path = resolve(from, snaps.get(snaps.len()-2).unwrap().clone());
    let from_scan = persist::load_snapshot(&from_path)?;
    let to_scan = persist::load_snapshot(&to_path)?;
    let diff = compute(&from_scan, &to_scan, file_label(&from_path), file_label(&to_path));
    Ok((diff, from_scan, to_scan))
}

pub fn enrich_with_unified_snippets(mut diff: DiffResult, from_scan: &ScanSummary, to_scan: &ScanSummary, max_file_size: usize, context: usize) -> DiffResult {
    use std::collections::HashMap;
    let mut map_from: HashMap<&str, &FileEntry> = HashMap::new();
    for f in &from_scan.files { map_from.insert(&f.path, f); }
    let mut map_to: HashMap<&str, &FileEntry> = HashMap::new();
    for f in &to_scan.files { map_to.insert(&f.path, f); }
    let mut unified: Vec<FileUnifiedDiff> = Vec::new();
    for m in &diff.modified {
        if m.old_size as usize > max_file_size || m.new_size as usize > max_file_size { continue; }
        let old_entry = map_from.get(m.path.as_str()).and_then(|e| e.snippet.as_ref());
        let new_entry = map_to.get(m.path.as_str()).and_then(|e| e.snippet.as_ref());
        let (Some(old_text), Some(new_text)) = (old_entry, new_entry) else { continue; };
        let hunks = build_unified_hunks(old_text, new_text, context);
        if !hunks.is_empty() { unified.push(FileUnifiedDiff { path: m.path.clone(), hunks }); }
    }
    if !unified.is_empty() { diff.unified = Some(unified); }
    diff
}

fn build_unified_hunks(old: &str, new: &str, context: usize) -> Vec<DiffHunk> {
    // Simple LCS-based diff using dynamic programming (optimized for short snippets)
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    // If huge, fallback
    if old_lines.len() + new_lines.len() > 20_000 { return vec![DiffHunk { header: format!("@@ large change {} -> {} lines @@", old_lines.len(), new_lines.len()), lines: vec!["(diff omitted due to size)".into()] }]; }
    // Build LCS table
    let m = old_lines.len();
    let n = new_lines.len();
    let mut dp = vec![vec![0u16; n+1]; m+1];
    for i in (0..m).rev() { for j in (0..n).rev() { dp[i][j] = if old_lines[i] == new_lines[j] { dp[i+1][j+1] + 1 } else { dp[i+1][j].max(dp[i][j+1]) }; } }
    // Backtrack to produce diff sequence
    #[derive(Clone, Debug)] enum Edit<'a> { Equal(&'a str), Del(&'a str), Add(&'a str) }
    let mut edits: Vec<Edit> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < m && j < n { if old_lines[i] == new_lines[j] { edits.push(Edit::Equal(old_lines[i])); i+=1; j+=1; } else if dp[i+1][j] >= dp[i][j+1] { edits.push(Edit::Del(old_lines[i])); i+=1; } else { edits.push(Edit::Add(new_lines[j])); j+=1; } }
    while i < m { edits.push(Edit::Del(old_lines[i])); i+=1; }
    while j < n { edits.push(Edit::Add(new_lines[j])); j+=1; }
    // Group into hunks with context
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut current: Vec<(usize,String)> = Vec::new(); // (old_line_no or 0, formatted)
    let mut old_line_no = 1usize; let mut new_line_no = 1usize;
    let mut hunk_old_start = 0usize; let mut hunk_new_start = 0usize;
    let mut in_hunk = false; let mut old_count = 0usize; let mut new_count = 0usize;
    for (idx, e) in edits.iter().enumerate() {
        let is_change = !matches!(e, Edit::Equal(_));
        if is_change && !in_hunk { // start hunk
            in_hunk = true; hunk_old_start = old_line_no; hunk_new_start = new_line_no;
            // include previous context lines
            let start_ctx = current.len().saturating_sub(context);
            let ctx_slice = &current[start_ctx..];
            let pre_lines: Vec<(usize,String)> = ctx_slice.to_vec();
            current = pre_lines;
            old_count = current.iter().filter(|(_, l)| l.starts_with(' ') || l.starts_with('-')).count();
            new_count = current.iter().filter(|(_, l)| l.starts_with(' ') || l.starts_with('+')).count();
        }
        match e {
            Edit::Equal(line) => {
                if in_hunk { current.push((old_line_no, format!(" {}", line))); old_count+=1; new_count+=1; }
                // if equal and hunk active, may need to end if too much trailing context
                if in_hunk {
                    // Peek ahead: if next changes farther than context, end hunk
                    let next_change_dist = edits.iter().skip(idx+1).position(|x| !matches!(x, Edit::Equal(_)));
                    if let Some(dist) = next_change_dist { if dist > context { // finalize
                        // Trim extra context > requested
                        // we allow current to carry extra; simplicity skip trimming
                        let header = format!("@@ -{},{} +{},{} @@", hunk_old_start, old_count, hunk_new_start, new_count);
                        hunks.push(DiffHunk { header, lines: current.iter().map(|(_,l)| l.clone()).collect() });
                        current = Vec::new(); in_hunk = false; old_count=0; new_count=0;
                    }} else { // end at file end
                        // handled after loop
                    }
                }
                old_line_no+=1; new_line_no+=1;
            }
            Edit::Del(line) => { current.push((old_line_no, format!("-{}", line))); old_line_no+=1; old_count+=1; }
            Edit::Add(line) => { current.push((old_line_no.saturating_sub(1), format!("+{}", line))); new_line_no+=1; new_count+=1; }
        }
    }
    if in_hunk && !current.is_empty() { let header = format!("@@ -{},{} +{},{} @@", hunk_old_start, old_count, hunk_new_start, new_count); hunks.push(DiffHunk { header, lines: current.into_iter().map(|(_,l)| l).collect() }); }
    hunks
}
