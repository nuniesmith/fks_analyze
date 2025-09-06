use std::{fs, process::Command, path::PathBuf, thread, time::Duration};

// Tests for snapshot meta, prune dry-run, and stats output

#[test]
fn snapshot_meta_and_prune_dry_run() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(root.join("a.txt"), "one").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    // create three snapshots (two changed files)
    for i in 0..3 { if i>0 { fs::write(root.join(format!("a{i}.txt")), format!("file{i}" )).unwrap(); }
        let out = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--save", "--compress", "true", "--content-bytes", "32"]).output().unwrap(); assert!(out.status.success()); thread::sleep(Duration::from_millis(1100)); }
    // stats json limited
    let stats = Command::new(&bin).args(["--root", root.to_str().unwrap(), "stats", "--format", "json", "--limit", "2"]).output().unwrap();
    assert!(stats.status.success());
    let stdout = String::from_utf8(stats.stdout).unwrap();
    assert!(stdout.contains("overall_compression_ratio"));
    // dry run prune keep 1 (should list at least one removal)
    let prune = Command::new(&bin).args(["--root", root.to_str().unwrap(), "prune", "--keep", "1", "--dry-run"]).output().unwrap();
    assert!(prune.status.success());
    let pstdout = String::from_utf8(prune.stdout).unwrap();
    assert!(pstdout.contains("Dry run: would remove"));
}
