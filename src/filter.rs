use globset::{GlobBuilder, GlobSetBuilder};
use anyhow::Result;

#[allow(dead_code)]
pub struct PathFilter { set: globset::GlobSet }

#[allow(dead_code)]
impl PathFilter {
    pub fn new(deny: &[&str]) -> Result<Self> { 
        let mut b = GlobSetBuilder::new();
        for d in deny { b.add(GlobBuilder::new(d).case_insensitive(true).build()?); }
        Ok(Self { set: b.build()? })
    }
    pub fn is_denied(&self, rel: &str) -> bool { self.set.is_match(rel) }
}

#[allow(dead_code)]
pub fn default_filter() -> PathFilter { 
    PathFilter::new(&["**/.git/**","**/target/**","**/node_modules/**","**/dist/**","**/build/**","**/venv/**","**/shared/**","shared_repos/**/shared_templates/**"]).expect("filter")
}
