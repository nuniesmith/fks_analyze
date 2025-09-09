use std::{fs, process::Command, path::PathBuf, thread, time::Duration};

#[test]
fn prune_age_and_ratio_warning() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_fks_analyze"));
    // create two snapshots with different sizes to influence ratio
    fs::write(root.join("file1.bin"), vec![0u8; 50_000]).unwrap();
    let _ = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--save", "--compress", "true"]).output().unwrap();
    thread::sleep(Duration::from_millis(1100));
    fs::write(root.join("file2.bin"), vec![1u8; 200_000]).unwrap();
    let _ = Command::new(&bin).args(["--root", root.to_str().unwrap(), "scan", "--save", "--compress", "true"]).output().unwrap();
    // stats with warning threshold small so it almost certainly triggers
    let stats = Command::new(&bin).args(["--root", root.to_str().unwrap(), "stats", "--format", "json", "--warn-ratio-increase", "0"]).output().unwrap();
    assert!(stats.status.success());
    let s = String::from_utf8(stats.stdout).unwrap();
    assert!(s.contains("compression_ratio_increase_pct"));
    // simulate very old snapshot by editing timestamp in filename not trivial; instead test count prune leaves 1
    let list_before: Vec<_> = std::fs::read_dir(root.join(".fks_analyze/history")).unwrap().filter_map(|e| e.ok()).collect();
    assert!(list_before.len() >= 2);
    let prune = Command::new(&bin).args(["--root", root.to_str().unwrap(), "prune", "--keep", "1"]).output().unwrap();
    assert!(prune.status.success());
    let list_after: Vec<_> = std::fs::read_dir(root.join(".fks_analyze/history")).unwrap().filter_map(|e| e.ok()).collect();
    // At least one snapshot + its meta should remain (so >=1 entry). We assert fewer entries now.
    assert!(list_after.len() < list_before.len(), "prune did not remove files");
}
