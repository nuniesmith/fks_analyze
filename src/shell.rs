use anyhow::Result;
use std::process::Command;
use std::path::PathBuf;
use tracing::{info, debug, warn};

#[derive(Debug)]
pub struct ScriptOptions {
    pub root: String,
    pub full: bool,
    pub lint: bool,
    pub analyze_type: Option<String>,
    pub output: Option<String>,
    pub exclude: Option<String>,
    pub skip_shared: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct ScriptReports {
    pub output_dir: Option<String>,
    pub used_fallback: bool,
}

#[allow(dead_code)]
pub fn run_analyze_script(opts: ScriptOptions) -> Result<ScriptReports> {
    let script = find_script(&opts.root)?;
    info!(target="analyze_script", path=%script.display(), root=%opts.root, "Using analyze script");
    // Inner closure to build and run command; returns (stdout, stderr, status_ok)
    let run_once = |forced_output: Option<&str>| -> Result<(String,String,bool)> {
        let mut cmd = Command::new("bash");
        let mut args = vec![script.to_string_lossy().to_string()];
        if opts.full { args.push("--full".into()); }
        if opts.lint { args.push("--lint".into()); }
        if let Some(out) = forced_output.or(opts.output.as_deref()) { args.push(format!("--output={out}")); }
        if let Some(ex) = &opts.exclude { args.push(format!("--exclude={ex}")); }
        if opts.skip_shared { args.push("--skip-shared".into()); }
        if let Some(t) = &opts.analyze_type { args.push(format!("--type={t}")); }
        args.push(opts.root.clone());
        debug!(target="analyze_script", forced_output=?forced_output, ?args, "Invoking analysis script");
        let child = cmd.args(args).current_dir(&opts.root).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn()?;
        let out = child.wait_with_output()?;
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        Ok((stdout, stderr, out.status.success()))
    };

    // Pre-detect if target root appears writable; if not, go straight to fallback output
    let root_writable = {
        let test_path = std::path::Path::new(&opts.root).join(".__analyze_writable_test__");
        match std::fs::File::create(&test_path) { Ok(_) => { let _ = std::fs::remove_file(&test_path); true }, Err(_) => false }
    };
    let mut used_fallback_path: Option<String> = None;
    let mut used_fallback = false;
    // Helper to build fallback dir path
    let build_fallback = || -> String {
        let service = std::path::Path::new(&opts.root).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "root".into());
        let base = std::env::var("ANALYZE_TMP_BASE").unwrap_or_else(|_| "/tmp/fks_analysis".into());
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        format!("{base}/{service}_{ts}")
    };
    if !root_writable && opts.output.is_none() {
        let fb = build_fallback();
        let _ = std::fs::create_dir_all(&fb);
    used_fallback_path = Some(fb.clone());
    used_fallback = true;
        info!(target="analyze_script", fallback=%fb, reason="root_read_only_precheck", "Using fallback output dir upfront");
    }
    let (mut stdout, mut stderr, mut ok) = run_once(used_fallback_path.as_deref())?;
    if !ok {
        let mut diag = if stderr.trim().is_empty() { stdout.trim().to_string() } else { stderr.trim().to_string() };
        // Detect read-only filesystem mkdir failure and retry with /tmp base
        let readonly = diag.contains("Read-only file system") || diag.contains("read-only file system");
        if readonly {
            let fallback_dir = build_fallback();
            if let Err(e) = std::fs::create_dir_all(&fallback_dir) { warn!(target="analyze_script", fallback=%fallback_dir, ?e, "Failed to pre-create fallback output dir"); }
            warn!(target="analyze_script", fallback=%fallback_dir, err=%diag, "Retrying analysis with writable temp output");
            let res = run_once(Some(&fallback_dir))?;
            stdout = res.0; stderr = res.1; ok = res.2; diag = if stderr.trim().is_empty() { stdout.trim().to_string() } else { stderr.trim().to_string() };
            used_fallback = true;
            if !ok {
                warn!(target="analyze_script", err=%diag, "Script failed after fallback retry");
                return Err(anyhow::anyhow!("analyze_codebase.sh failed (fallback): {diag}"));
            }
        } else {
            warn!(target="analyze_script", err=%diag, "Script failed");
            return Err(anyhow::anyhow!("analyze_codebase.sh failed: {diag}"));
        }
    }
    debug!(target="analyze_script", stdout_preview=stdout.lines().take(12).collect::<Vec<_>>().join(" | "), "Script stdout (first lines)");
    let output_dir = stdout.lines().find_map(|l| {
        if l.contains("Reports will be in:") { l.split_once(": ").map(|(_,r)| r.trim().to_string()) } else { None }
    });
    info!(target="analyze_script", output_dir=?output_dir, used_fallback=%used_fallback, "Analysis script completed");
    Ok(ScriptReports { output_dir, used_fallback })
}

#[allow(dead_code)]
fn find_script(root: &str) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("FKS_ANALYZE_SCRIPT") {
        let path = PathBuf::from(&p);
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(&path) {
                info!(target="analyze_script", provided_env=%p, size=meta.len(), is_file=%meta.is_file(), "Using script from FKS_ANALYZE_SCRIPT");
            } else {
                info!(target="analyze_script", provided_env=%p, "Using script from FKS_ANALYZE_SCRIPT (metadata unavailable)");
            }
            return Ok(path);
        } else {
            warn!(target="analyze_script", provided_env=%p, "FKS_ANALYZE_SCRIPT path does not exist, continuing to search");
        }
    }
    // Allow pointing to a directory containing the script
    if let Ok(dir) = std::env::var("FKS_ANALYZE_SCRIPT_DIR") {
        let p = PathBuf::from(dir).join("analyze_codebase.sh");
        if p.exists() { return Ok(p); }
    }
    // Candidate search order
    let candidates = [
        "shared/scripts/utils/analyze_codebase.sh",
        "shared/scripts/devtools/analysis/analyze_codebase.sh",
        "shared_scripts/utils/analyze_codebase.sh",
        "shared_scripts/devtools/analysis/analyze_codebase.sh",
        "shared_repos/shared_scripts/utils/analyze_codebase.sh",
        "shared_repos/shared_scripts/devtools/analysis/analyze_codebase.sh",
        "../shared/scripts/utils/analyze_codebase.sh",
        "../shared/scripts/devtools/analysis/analyze_codebase.sh",
    ];
    let abs_candidates = [
        "/app/shared/scripts/utils/analyze_codebase.sh",
        "/app/shared/scripts/devtools/analysis/analyze_codebase.sh",
    "/app/analyze_codebase.sh",
    ];
    let mut tried: Vec<PathBuf> = Vec::new();
    for rel in candidates {
        let candidate = PathBuf::from(root).join(rel);
        if candidate.exists() {
            if let Ok(meta) = std::fs::metadata(&candidate) { debug!(target="analyze_script", found=%candidate.display(), size=meta.len(), is_file=%meta.is_file(), "Found analyze script candidate (relative)"); } else { debug!(target="analyze_script", found=%candidate.display(), "Found analyze script candidate (relative)"); }
            return Ok(candidate);
        } else { tried.push(candidate); }
    }
    for abs in abs_candidates {
        let p = PathBuf::from(abs);
        if p.exists() {
            if let Ok(meta) = std::fs::metadata(&p) { debug!(target="analyze_script", found=%p.display(), size=meta.len(), is_file=%meta.is_file(), "Found analyze script candidate (absolute)"); } else { debug!(target="analyze_script", found=%p.display(), "Found analyze script candidate (absolute)"); }
            return Ok(p);
        } else { tried.push(p); }
    }
    // Diagnostic: list a few entries under /app/shared if present to aid debugging volume issues
    if let Ok(entries) = std::fs::read_dir("/app/shared") {
        let mut names: Vec<String> = entries.flatten().take(30).map(|e| e.file_name().to_string_lossy().to_string()).collect();
        names.sort();
        debug!(target="analyze_script", shared_listing=?names, "Listing /app/shared (truncated)");
    }
    warn!(target="analyze_script", tried=?tried, root=%root, "Analyze script not found in any candidate paths");
    Err(anyhow::anyhow!("analyze_codebase.sh not found. Set FKS_ANALYZE_SCRIPT or FKS_ANALYZE_SCRIPT_DIR"))
}
