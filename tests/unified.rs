use std::{fs, process::Command, path::PathBuf, thread, time::Duration};

#[test]
fn unified_diff_skips_large_files_over_threshold() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let large_size = 8_000usize; // bytes
    let original = "A".repeat(large_size);
    fs::write(root.join("big.txt"), &original).unwrap();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    // First snapshot with snippet capture
    let out1 = Command::new(&bin).args([
        "--root", root.to_str().unwrap(),
        "scan","--save","--content-bytes","16000","--compress","false"
    ]).output().unwrap();
    assert!(out1.status.success(), "first scan failed: {:?}", out1);
    // Modify file (change some bytes)
    let mut modified = original.clone();
    modified.replace_range(0..10, "BBBBBBBBBB");
    fs::write(root.join("big.txt"), &modified).unwrap();
    thread::sleep(Duration::from_millis(1100)); // ensure distinct timestamp for snapshot name ordering
    let out2 = Command::new(&bin).args([
        "--root", root.to_str().unwrap(),
        "scan","--save","--content-bytes","16000","--compress","false"
    ]).output().unwrap();
    assert!(out2.status.success(), "second scan failed: {:?}", out2);
    // Run diff with a unified max smaller than file size so unified diff should be skipped
    let diff_out = Command::new(&bin).args([
        "--root", root.to_str().unwrap(),
        "diff","--unified","--unified-max","1000","--unified-context","3","--format","json"
    ]).output().unwrap();
    assert!(diff_out.status.success(), "diff failed: {:?}", diff_out);
    let json = String::from_utf8(diff_out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    // Ensure file is listed as modified
    let modified_arr = v.get("modified").and_then(|m| m.as_array()).expect("modified array");
    assert!(modified_arr.iter().any(|m| m.get("path").and_then(|p| p.as_str()) == Some("big.txt")), "big.txt not in modified list: {json}");
    // unified should be null OR not contain big.txt
    if let Some(unified) = v.get("unified") {
        if !unified.is_null() {
            let arr = unified.as_array().expect("unified array");
            assert!(arr.iter().all(|f| f.get("path").and_then(|p| p.as_str()) != Some("big.txt")), "Unified diff unexpectedly present for big.txt (> threshold). JSON: {json}");
        }
    }
}
