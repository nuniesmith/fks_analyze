use std::{fs, process::Command, path::PathBuf};

#[test]
fn pipeline_fail_on_delta_triggers() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Baseline with tests present and some source files (so removing tests later creates High severity)
    fs::write(root.join("README.md"), "Docs").unwrap();
    std::fs::create_dir_all(root.join("tests")).unwrap();
    fs::write(root.join("tests/t.rs"), "#[test] fn ok(){assert!(true);}"
    ).unwrap();
    for i in 0..5 { fs::write(root.join(format!("s{i}.rs")), "fn x(){}" ).unwrap(); }
    // First pipeline run saving findings
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    assert!(Command::new(&bin).args(["--root", root.to_str().unwrap(), "pipeline", "--save-findings"]).status().unwrap().success());
    std::thread::sleep(std::time::Duration::from_millis(1100));
    // Remove tests to create new high severity on second run
    std::fs::remove_dir_all(root.join("tests")).unwrap();
    for i in 0..4 { fs::write(root.join(format!("m{i}.rs")), "fn main(){}" ).unwrap(); }
    let status = Command::new(&bin).args(["--root", root.to_str().unwrap(), "pipeline", "--save-findings", "--fail-on-delta", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(2), "Expected delta gating to fail when high count increases");
}

#[test]
fn pipeline_fail_on_delta_noop_with_single_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    // Only one run
    let status = Command::new(&bin).args(["--root", root.to_str().unwrap(), "pipeline", "--save-findings", "--fail-on-delta", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(0), "Single snapshot should not trigger delta gating");
}
