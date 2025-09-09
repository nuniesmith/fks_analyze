use crate::scan::{scan_repo_opts, scan_repo_with_ignore, ScanSummary};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::fs;
use std::path::{PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct MultiScanSummary {
    pub root: String,
    pub repos: Vec<RepoScan>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RepoScan {
    pub name: String,
    pub path: String,
    pub files: usize,
    pub size: u64,
    pub scan: ScanSummary,
}

pub fn scan_all(root: &str, content_bytes: usize, skip_shared: bool, include_shared_repos: bool, ignore: &[String]) -> Result<MultiScanSummary> {
    let root_path = PathBuf::from(root);
    let parent = root_path.parent().unwrap_or(&root_path);
    let mut repos: Vec<RepoScan> = Vec::new();
    for entry in fs::read_dir(parent)? { if let Ok(e)=entry { let p = e.path(); if p.is_dir() { if let Some(name)=p.file_name().and_then(|s| s.to_str()) { 
        if name.starts_with("fks_") && name != "fks_analyze" { 
            let scan = if ignore.is_empty() { scan_repo_opts(p.to_string_lossy().as_ref(), content_bytes, skip_shared)? } else { scan_repo_with_ignore(p.to_string_lossy().as_ref(), content_bytes, skip_shared, ignore)? };
            repos.push(RepoScan { name: name.to_string(), path: p.to_string_lossy().to_string(), files: scan.total_files, size: scan.total_size, scan });
        }
        if include_shared_repos && name == "shared_repos" { 
            // treat each shared_* inside shared_repos as its own logical repo (excluding shared_templates duplication)
            for shared_entry in fs::read_dir(p)? { if let Ok(se)=shared_entry { let sp = se.path(); if sp.is_dir() { if let Some(sname)=sp.file_name().and_then(|s| s.to_str()) { if sname.starts_with("shared_") { 
                let scan = if ignore.is_empty() { scan_repo_opts(sp.to_string_lossy().as_ref(), content_bytes, true)? } else { scan_repo_with_ignore(sp.to_string_lossy().as_ref(), content_bytes, true, ignore)? }; // skip its internal shared/
                repos.push(RepoScan { name: sname.to_string(), path: sp.to_string_lossy().to_string(), files: scan.total_files, size: scan.total_size, scan });
            }}}}}
        }
    } } } }
    repos.sort_by(|a,b| a.name.cmp(&b.name));
    Ok(MultiScanSummary { root: parent.to_string_lossy().to_string(), repos })
}