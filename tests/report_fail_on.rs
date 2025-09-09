use std::{fs, process::Command, path::PathBuf};

#[test]
fn report_fail_on_triggers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // create rust files without tests or README to trigger findings
    for i in 0..4 { fs::write(root.join(format!("mod{i}.rs")), "fn main(){}" ).unwrap(); }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "report", "--fail-on", "medium", "--format", "json"]) // json to ensure branch also works
        .status().unwrap();
    assert_eq!(status.code(), Some(2), "Report should exit 2 on medium threshold");
}

#[test]
fn report_fail_on_passes_when_below() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(root.join("README.md"), "Docs").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "report", "--fail-on", "high", "--format", "json"]) // expect no high severity
        .status().unwrap();
    assert_eq!(status.code(), Some(0), "Report should exit 0 when high not present");
}
