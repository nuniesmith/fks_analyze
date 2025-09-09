#[cfg(feature="semantic")]
use anyhow::Result;
#[cfg(feature="semantic")]
use tantivy::{schema::*, Index, IndexWriter, collector::TopDocs, query::QueryParser, doc};
#[cfg(feature="semantic")]
use std::{path::PathBuf, fs};
#[cfg(feature="semantic")]
use crate::scan;

#[cfg(feature="semantic")]
pub struct SemanticIndex { pub dir: PathBuf, pub index: Index, pub path_field: Field, pub content_field: Field }

#[cfg(feature="semantic")]
pub fn build_semantic(root: &str, content_bytes: usize, force: bool) -> Result<SemanticIndex> {
    let dir = PathBuf::from(root).join(".fks_analyze/semantic");
    fs::create_dir_all(&dir)?;
    let mut schema_builder = Schema::builder();
    let path_field = schema_builder.add_text_field("path", TEXT | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT);
    let schema = schema_builder.build();
    let index_path = dir.join("index");
    if force && index_path.exists() { let _ = fs::remove_dir_all(&index_path); }
    let index = if index_path.exists() { Index::open_in_dir(&index_path)? } else { Index::create_in_dir(&index_path, schema.clone())? };
    let mut writer: IndexWriter = index.writer(50_000_000)?; // 50MB heap
    // Perform a scan (with snippets) to avoid rereading all files if already in memory
    let scan = scan::scan_repo(root, content_bytes)?;
    let max_bytes: usize = std::env::var("FKS_SEMANTIC_MAX_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    for f in scan.files {
        // Skip binary-ish large files
        if f.size as usize > max_bytes { continue; }
        let content = if let Some(snippet) = f.snippet { snippet } else {
            if f.size == 0 { String::new() } else { fs::read_to_string(PathBuf::from(root).join(&f.path)).unwrap_or_default() }
        };
        if content.trim().is_empty() { continue; }
    // Build document with macro (avoids direct Document type dependency changes across versions)
    writer.add_document(doc!(path_field => f.path.clone(), content_field => content))?;
    }
    writer.commit()?;
    Ok(SemanticIndex { dir, index, path_field, content_field })
}

#[cfg(feature="semantic")]
pub fn open_semantic(root: &str) -> Result<Option<SemanticIndex>> {
    let dir = PathBuf::from(root).join(".fks_analyze/semantic/index");
    if !dir.exists() { return Ok(None); }
    let index = Index::open_in_dir(&dir)?;
    let schema = index.schema();
    let path_field = schema.get_field("path").unwrap();
    let content_field = schema.get_field("content").unwrap();
    Ok(Some(SemanticIndex { dir: dir.parent().unwrap().to_path_buf(), index, path_field, content_field }))
}

#[cfg(feature="semantic")]
pub fn semantic_search(root: &str, query: &str, limit: usize) -> Result<Vec<(String, f32, String)>> {
    if let Some(idx) = open_semantic(root)? {
        let reader = idx.index.reader()?;
        let searcher = reader.searcher();
        let qp = QueryParser::for_index(&idx.index, vec![idx.content_field]);
        let q = qp.parse_query(query)?;
        let top = searcher.search(&q, &TopDocs::with_limit(limit))?;
        let mut out = Vec::new();
        for (score, addr) in top {
            let retrieved = searcher.doc::<tantivy::schema::TantivyDocument>(addr)?;
            if let Some(pv) = retrieved.get_first(idx.path_field) {
                let owned: tantivy::schema::OwnedValue = pv.into();
                if let tantivy::schema::OwnedValue::Str(path) = owned {
                    let content = std::fs::read_to_string(PathBuf::from(root).join(&path)).unwrap_or_default();
                    let snippet = make_snippet(&content, query, 240);
                    out.push((path, score, snippet));
                }
            }
        }
        Ok(out)
    } else {
        Ok(Vec::new())
    }
}

#[cfg(feature="semantic")]
fn make_snippet(content: &str, query: &str, max: usize) -> String {
    let terms: Vec<&str> = query.split_whitespace().collect();
    for t in &terms {
        if let Some(pos) = content.to_lowercase().find(&t.to_lowercase()) {
            let start = pos.saturating_sub(60); let end = (pos + t.len() + 60).min(content.len());
            let mut snip = content[start..end].to_string();
            snip = snip.replace('\n', " ");
            if snip.len() > max { snip.truncate(max); }
            return snip;
        }
    }
    content.chars().take(max).collect()
}
