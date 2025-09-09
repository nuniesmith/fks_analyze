use std::fs; use std::process::Command; use std::path::PathBuf;

#[test]
fn scan_ignore_patterns_excludes_files() {
    let tmp = tempfile::tempdir().unwrap(); let root = tmp.path();
    fs::write(root.join("keep.txt"), "a").unwrap();
    fs::write(root.join("ignore.log"), "b").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let out = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--ignore", "**/*.log"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("keep.txt"));
    assert!(!stdout.contains("ignore.log"), "ignore.log should be excluded: {stdout}");
}