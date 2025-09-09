use anyhow::Result;
use notify::{RecommendedWatcher, RecursiveMode, Watcher, Config};
use std::{sync::mpsc::channel, time::{Duration, Instant}, path::PathBuf};
use crate::{scan, persist, diff};

#[allow(dead_code)]
pub fn run_watch(root: String, content_bytes: usize) -> Result<()> {
    println!("Watching {} (Ctrl+C to stop)", root);
    let (tx, rx) = channel();
    let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
    watcher.watch(PathBuf::from(&root).as_path(), RecursiveMode::Recursive)?;
    let mut last = Instant::now();
    loop {
        match rx.recv() { Ok(_event) => {
            if last.elapsed() < Duration::from_millis(400) { continue; }
            last = Instant::now();
            let scan_data = scan::scan_repo(&root, content_bytes)?;
            let _snap = persist::save_snapshot(&root, &scan_data)?;
            if let Ok(d) = diff::latest_diff(&root) { println!("{}", d.render_text()); }
        } Err(_) => break }
    }
    Ok(())
}
