use std::{fs, process::Command, path::PathBuf};

#[test]
fn pipeline_json_outputs_valid() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(root.join("README.md"), "Docs").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let output = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "pipeline", "--format", "json"]) 
        .output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v.get("tasks").is_some(), "tasks field missing");
    assert!(v.get("counts").is_some(), "counts field missing");
}
