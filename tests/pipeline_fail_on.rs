use std::{fs, process::Command, path::PathBuf};

#[test]
fn pipeline_fail_on_triggers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for i in 0..6 { fs::write(root.join(format!("mod{i}.rs")), "fn main(){}" ).unwrap(); }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "pipeline", "--fail-on", "medium"]).status().unwrap();
    assert_eq!(status.code(), Some(2), "Pipeline should exit 2 for medium findings");
}

#[test]
fn pipeline_fail_on_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(root.join("README.md"), "Docs").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "pipeline", "--fail-on", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(0), "Pipeline should pass when high severity absent");
}
