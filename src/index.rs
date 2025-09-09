use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::{path::PathBuf, fs};
use crate::scan::scan_repo;

#[derive(Debug, Serialize, Deserialize)]
struct IndexedFile { path: String, content: String }

#[derive(Debug)]
pub struct SearchHit { pub path: String, pub score: f32, pub snippet: Option<String> }

pub struct SearchIndex { pub dir: PathBuf, files: Vec<IndexedFile> }

pub fn open_or_build(root: &str, content_bytes: usize) -> Result<SearchIndex> {
    let dir = PathBuf::from(root).join(".fks_analyze/index");
    fs::create_dir_all(&dir)?;
    let data_path = dir.join("files.json");
    if data_path.exists() { 
        let data = fs::read(data_path)?; 
        let files: Vec<IndexedFile> = serde_json::from_slice(&data)?; 
        return Ok(SearchIndex { dir, files });
    }
    build_index(root, &dir, content_bytes).map(|files| SearchIndex { dir, files })
}

fn build_index(root: &str, dir: &PathBuf, content_bytes: usize) -> Result<Vec<IndexedFile>> {
    let scan = scan_repo(root, if content_bytes==0 { 4096 } else { content_bytes })?;
    let mut out = Vec::new();
    for f in scan.files { if let Some(mut snippet)=f.snippet { 
        if snippet.len()>4096 { snippet.truncate(4096); }
        out.push(IndexedFile { path: f.path, content: snippet });
    } }
    fs::write(dir.join("files.json"), serde_json::to_vec(&out)?)?;
    Ok(out)
}

pub fn rebuild(root: &str, content_bytes: usize) -> Result<SearchIndex> {
    let dir = PathBuf::from(root).join(".fks_analyze/index");
    if dir.exists() { let data_path = dir.join("files.json"); let _ = std::fs::remove_file(data_path); }
    std::fs::create_dir_all(&dir)?;
    build_index(root, &dir, content_bytes).map(|files| SearchIndex { dir, files })
}

impl SearchIndex {
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let q = query.to_lowercase();
        let mut hits = Vec::new();
        for file in &self.files { if let Some(pos) = file.content.to_lowercase().find(&q) { 
            let start = pos.saturating_sub(40); let end = (pos+q.len()+40).min(file.content.len());
            let snippet = file.content[start..end].to_string();
            // naive score: inverse of position
            let score = 1.0f32 / (1.0 + pos as f32);
            hits.push(SearchHit { path: file.path.clone(), score, snippet: Some(snippet) });
        }}
        hits.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
        hits.truncate(limit);
        Ok(hits)
    }
    pub fn len(&self) -> usize { self.files.len() }
}
