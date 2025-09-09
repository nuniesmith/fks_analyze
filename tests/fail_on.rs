use std::{fs, process::Command, path::PathBuf};

#[test]
fn fail_on_exits_nonzero_when_threshold_met() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // create a situation likely to produce a Medium finding: missing README + add many rust files but no tests
    for i in 0..6 { fs::write(root.join(format!("mod{i}.rs")), "fn main(){}" ).unwrap(); }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "findings", "--fail-on", "medium"]).status().unwrap();
    assert_eq!(status.code(), Some(2), "Expected exit code 2 when medium findings present: {:?}", status);
}

#[test]
fn fail_on_passes_when_below_threshold() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // only a README (should produce few/no higher severity findings)
    fs::write(root.join("README.md"), "Docs").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let status = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "findings", "--fail-on", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(0), "High severity not present, should exit 0");
}
