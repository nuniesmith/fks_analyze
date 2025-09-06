use std::{fs, process::Command, path::PathBuf, thread, time::Duration};

#[test]
fn report_includes_compression_trend() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::write(root.join("first.txt"), "alpha").unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    // snapshot 1
    let out1 = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--save", "--compress", "true", "--content-bytes", "64"]).output().unwrap();
    assert!(out1.status.success());
    thread::sleep(Duration::from_millis(1100));
    // change and snapshot 2
    fs::write(root.join("second.txt"), "beta").unwrap();
    let out2 = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--save", "--compress", "true", "--content-bytes", "64"]).output().unwrap();
    assert!(out2.status.success());
    // generate markdown report
    let rep = Command::new(&bin).args(["--root", root.to_str().unwrap(), "report", "--format", "md"]) .output().unwrap();
    assert!(rep.status.success());
    let stdout = String::from_utf8(rep.stdout).unwrap();
    assert!(stdout.contains("Compression Trend"), "Report missing Compression Trend section: {}", stdout);
}
