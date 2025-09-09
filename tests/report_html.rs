use std::{fs, process::Command, path::PathBuf};

#[test]
fn report_html_outputs_basic_structure() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // create some rust files to generate findings
    for i in 0..3 { fs::write(root.join(format!("m{i}.rs")), "fn main(){}" ).unwrap(); }
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    let output = Command::new(&bin)
        .args(["--root", root.to_str().unwrap(), "report", "--format", "html"])
        .output().unwrap();
    assert!(output.status.success());
    let s = String::from_utf8_lossy(&output.stdout);
    assert!(s.contains("<html"), "Expected html tag");
    assert!(s.contains("Findings Summary"), "Expected findings summary section");
    assert!(s.contains("table"), "Expected a table tag");
}
