use std::fs; use std::process::Command; use std::path::PathBuf;

// Basic unit-like tests invoking library via binary since modules are mostly internal.

#[test]
fn scan_creates_summary() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // create a couple files
    fs::write(root.join("README.md"), "Hello world").unwrap();
    fs::write(root.join("main.rs"), "fn main(){}" ).unwrap();
    // run scan
    // Build binary once (assumes cargo test has already built); locate target/debug/fks_analyze
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let out = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan"]).output().unwrap();
    assert!(out.status.success(), "scan failed: {:?}", out);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("total_files"));
}

#[test]
fn diff_between_snapshots() {
    let tmp = tempfile::tempdir().unwrap(); let root = tmp.path();
    fs::write(root.join("a.txt"), "one").unwrap();
    // first snapshot
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let _ = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan","--save"]).output().unwrap();
    fs::write(root.join("b.txt"), "two").unwrap();
    // ensure filesystem timestamp difference (some filesystems have 1s granularity)
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let _ = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan","--save"]).output().unwrap();
    let out = Command::new(&bin).args(["--root", root.to_str().unwrap(), "diff"]).output().unwrap();
    if !out.status.success() {
        panic!("diff failed: status={:?} stderr={} stdout={}", out.status, String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    }
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("Added:"), "unexpected diff output: {}", stdout);
}

#[test]
fn suggest_outputs_prompt_text() {
    let tmp = tempfile::tempdir().unwrap(); let root = tmp.path();
    fs::write(root.join("main.rs"), "fn main(){}" ).unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let out = Command::new(&bin).args(["--root", root.to_str().unwrap(), "suggest"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("Next prioritized tasks"));
}
