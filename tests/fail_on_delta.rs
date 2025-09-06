use std::{fs, process::Command, path::PathBuf};

// Helper to run findings command
fn run_findings(root: &std::path::Path, extra: &[&str]) -> std::process::ExitStatus {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    Command::new(&bin).args(["--root", root.to_str().unwrap(), "findings"]).args(extra).status().unwrap()
}

#[test]
fn fail_on_delta_triggers_on_increase() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Baseline snapshot with tests present so no HIGH severity (RUST_TESTS_MISSING) finding
    fs::write(root.join("README.md"), "Docs").unwrap();
    fs::write(root.join("LICENSE"), "MIT").unwrap();
    std::fs::create_dir_all(root.join("tests")).unwrap();
    fs::write(root.join("tests/ok.rs"), "#[test] fn it_works() { assert_eq!(2+2,4); }").unwrap();
    fs::write(root.join("src_one.rs"), "fn a(){}" ).unwrap();
    assert!(run_findings(root, &["--save"]).success());
    std::thread::sleep(std::time::Duration::from_millis(1100));
    // Second snapshot: remove tests directory and add more rust files -> triggers HIGH severity (missing tests)
    std::fs::remove_dir_all(root.join("tests")).unwrap();
    for i in 0..5 { fs::write(root.join(format!("m{i}.rs")), "fn main(){}" ).unwrap(); }
    let status = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze")))
        .args(["--root", root.to_str().unwrap(), "findings", "--save", "--fail-on-delta", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(2), "Should exit 2 due to increased high counts");
}

#[test]
fn fail_on_delta_ignores_when_no_increase() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
        // Create enough rust files without tests to induce a single HIGH severity (missing tests)
        for i in 0..6 { fs::write(root.join(format!("m{i}.rs")), "fn main(){}" ).unwrap(); }
    assert!(run_findings(root, &["--save"]).success());
    std::thread::sleep(std::time::Duration::from_millis(1100));
        // Run again adding a couple more files; expect still single HIGH severity count
        for i in 6..8 { fs::write(root.join(format!("m{i}.rs")), "fn main(){}" ).unwrap(); }
    let status = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze")))
        .args(["--root", root.to_str().unwrap(), "findings", "--save", "--fail-on-delta", "high"]).status().unwrap();
    assert_eq!(status.code(), Some(0), "No increase in high severity count so delta gating should not fail");
}
