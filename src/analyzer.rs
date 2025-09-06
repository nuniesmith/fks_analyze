use crate::scan::ScanSummary;
use serde::{Serialize, Deserialize};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use chrono::{Utc, DateTime};
use once_cell::sync::Lazy;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Severity { Info, Low, Medium, High, Critical }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Finding {
    pub id: String,
    pub title: String,
    pub description: String,
    #[serde(skip_serializing_if="Option::is_none")] pub path: Option<String>,
    pub severity: Severity,
    #[serde(skip_serializing_if="Option::is_none")] pub remediation: Option<String>,
    #[serde(default, skip_serializing_if="Vec::is_empty")] pub tags: Vec<String>,
    #[serde(skip_serializing_if="Option::is_none")] pub kind: Option<FindingKind>,
    #[serde(skip_serializing_if="Option::is_none")] pub score: Option<f32>,
    #[serde(skip_serializing_if="Option::is_none")] pub created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub enum FindingKind { Documentation, Testing, Size, Maintenance, Hotspot, DocsGap }

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct FindingsReport {
    pub counts: BTreeMap<Severity, usize>,
    pub findings: Vec<Finding>,
    pub meta: BTreeMap<String, String>,
}

pub trait Analyzer: Send + Sync {
    fn name(&self) -> &'static str;
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding>;
}

pub struct CompositeAnalyzer { analyzers: Vec<Box<dyn Analyzer>> }

impl CompositeAnalyzer {
    pub fn new() -> Self { Self { analyzers: Vec::new() } }
    pub fn register<A: Analyzer + 'static>(mut self, a: A) -> Self { self.analyzers.push(Box::new(a)); self }
    pub fn run(&self, scan: &ScanSummary) -> FindingsReport {
        let mut findings: Vec<Finding> = Vec::new();
        for a in &self.analyzers {
            let name = a.name();
            let produced = a.analyze(scan);
            let count = produced.len();
            findings.extend(produced);
            if std::env::var("FKS_ANALYZER_TRACE").ok().as_deref() == Some("1") {
                eprintln!("analyzer:{} findings={} total={} ", name, count, findings.len());
            }
        }
        let mut counts: BTreeMap<Severity, usize> = BTreeMap::new();
        for f in &findings { *counts.entry(f.severity.clone()).or_insert(0) += 1; }
    let mut report = FindingsReport { counts, findings, meta: BTreeMap::new() };
    report.meta.insert("generated_at".into(), Utc::now().to_rfc3339());
    score_findings(&mut report);
    report
    }
}

// ---------------- Built‑in Analyzers ----------------

pub struct ReadmePresence;
impl Analyzer for ReadmePresence {
    fn name(&self) -> &'static str { "readme_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let has_readme = scan.files.iter().any(|f| f.path.eq_ignore_ascii_case("README.md"));
    if !has_readme { vec![Finding { id: "README_MISSING".into(), title: "Missing README.md".to_string(), description: "Repository lacks a README.md file.".to_string(), path: None, severity: Severity::Medium, remediation: Some("Add a README.md describing purpose, setup, and contributing.".to_string()), tags: vec!["docs".into(),"readme".into()], kind: Some(FindingKind::Documentation), score: None, created_at: Some(Utc::now().to_rfc3339()) }] } else { vec![] }
    }
}

pub struct TestCoverageHeuristic;
impl Analyzer for TestCoverageHeuristic {
    fn name(&self) -> &'static str { "test_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let rust_files = scan.files.iter().filter(|f| f.path.ends_with(".rs")).count();
        let rust_tests = scan.files.iter().filter(|f| f.path.starts_with("tests/") || f.path.contains("_test.rs")).count();
        if rust_files > 5 && rust_tests == 0 {
            return vec![Finding { id: "RUST_TESTS_MISSING".into(), title: "Missing Rust tests".to_string(), description: format!("Detected {rust_files} Rust source files but no test files."), path: None, severity: Severity::High, remediation: Some("Add tests under tests/ or *_test.rs for critical modules.".to_string()), tags: vec!["tests".into(),"rust".into()], kind: Some(FindingKind::Testing), score: None, created_at: Some(Utc::now().to_rfc3339()) }];
        }
        vec![]
    }
}

pub struct LargeFileDetector { pub threshold: u64 }
impl Analyzer for LargeFileDetector {
    fn name(&self) -> &'static str { "large_files" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
    scan.files.iter().filter(|f| f.size > self.threshold).map(|f| Finding { id: format!("LARGE_FILE::{}", f.path), title: "Large file".to_string(), description: format!("File {} is {} bytes (> {} threshold).", f.path, f.size, self.threshold), path: Some(f.path.clone()), severity: Severity::Low, remediation: Some("Consider splitting or compressing if this impacts performance or readability.".to_string()), tags: vec!["size".into()], kind: Some(FindingKind::Size), score: None, created_at: Some(Utc::now().to_rfc3339()) }).collect()
    }
}

pub struct TodoCommentDetector;
impl Analyzer for TodoCommentDetector {
    fn name(&self) -> &'static str { "todo_comments" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let re = Regex::new(r"(?i)TODO|FIXME|HACK|XXX").unwrap();
        let mut findings = Vec::new();
    for f in &scan.files { if let Some(snippet) = &f.snippet { if re.is_match(snippet) { findings.push(Finding { id: format!("TODO_IN::{}", f.path), title: "TODO/FIXME present".to_string(), description: format!("Maintenance marker found in {}", f.path), path: Some(f.path.clone()), severity: Severity::Info, remediation: Some("Address the TODO/FIXME and remove the marker.".to_string()), tags: vec!["todo".into(),"debt".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) }); } } }
        findings
    }
}

pub struct ExtensionDiversity;
impl Analyzer for ExtensionDiversity {
    fn name(&self) -> &'static str { "ext_diversity" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let mut ext_counts: HashMap<String, usize> = HashMap::new();
        for (ext, count) in &scan.counts_by_ext { ext_counts.insert(ext.clone(), *count); }
        if ext_counts.get("rs").copied().unwrap_or(0) > 0 && ext_counts.get("md").copied().unwrap_or(0) == 0 {
            return vec![Finding { id: "DOC_GAP".into(), title: "No markdown docs".to_string(), description: "Rust code present but no markdown documentation files detected.".to_string(), path: None, severity: Severity::Medium, remediation: Some("Add architecture and usage docs (.md).".to_string()), tags: vec!["docs".into(),"rust".into()], kind: Some(FindingKind::DocsGap), score: None, created_at: Some(Utc::now().to_rfc3339()) }];
        }
        vec![]
    }
}

// License presence analyzer
pub struct LicensePresence;
impl Analyzer for LicensePresence {
    fn name(&self) -> &'static str { "license_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let has_license = scan.files.iter().any(|f| {
            let p = f.path.to_lowercase();
            p == "license" || p == "license.txt" || p == "license.md" || p.starts_with("license-")
        });
        if !has_license { vec![Finding { id: "LICENSE_MISSING".into(), title: "Missing LICENSE".into(), description: "Project has no LICENSE file; clarify usage terms.".into(), path: None, severity: Severity::Low, remediation: Some("Add an OSS license file (e.g., MIT, Apache-2.0).".into()), tags: vec!["license".into(),"compliance".into()], kind: Some(FindingKind::Documentation), score: None, created_at: Some(Utc::now().to_rfc3339()) }] } else { vec![] }
    }
}

// CI workflow presence analyzer
pub struct CiWorkflowPresence;
impl Analyzer for CiWorkflowPresence {
    fn name(&self) -> &'static str { "ci_workflow_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let has_ci = scan.files.iter().any(|f| f.path.starts_with(".github/workflows/") && (f.path.ends_with(".yml") || f.path.ends_with(".yaml")));
        if !has_ci { vec![Finding { id: "CI_MISSING".into(), title: "Missing CI workflow".into(), description: "No GitHub Actions workflow files detected.".into(), path: None, severity: Severity::Medium, remediation: Some("Add CI (.github/workflows/*.yml) to run tests, lint, and security checks.".into()), tags: vec!["ci".into(),"workflow".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) }] } else { vec![] }
    }
}

// Python test heuristic analyzer
pub struct PythonTestHeuristic;
impl Analyzer for PythonTestHeuristic {
    fn name(&self) -> &'static str { "python_test_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let py_files = scan.files.iter().filter(|f| f.path.ends_with(".py") && !f.path.contains("/tests/")).count();
        let py_tests = scan.files.iter().filter(|f| f.path.contains("/tests/") && f.path.ends_with(".py")).count();
        if py_files > 10 && py_tests == 0 { return vec![Finding { id: "PY_TESTS_MISSING".into(), title: "Missing Python tests".into(), description: format!("Detected {py_files} Python files but no tests."), path: None, severity: Severity::High, remediation: Some("Add pytest tests under tests/ directory.".into()), tags: vec!["tests".into(),"python".into()], kind: Some(FindingKind::Testing), score: None, created_at: Some(Utc::now().to_rfc3339()) }]; }
        vec![]
    }
}

// JavaScript / TypeScript test heuristic analyzer
pub struct JsTestHeuristic;
impl Analyzer for JsTestHeuristic {
    fn name(&self) -> &'static str { "js_test_presence" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        let mut js_files = 0usize;
        let mut js_tests = 0usize;
        for f in &scan.files {
            let p = f.path.as_str();
            if p.contains("node_modules/") { continue; }
            // identify TS/JS source (skip type definition stubs)
            let is_code = p.ends_with(".js") || p.ends_with(".jsx") || (p.ends_with(".ts") && !p.ends_with(".d.ts")) || p.ends_with(".tsx");
            if !is_code { continue; }
            let filename = p.rsplit('/').next().unwrap_or(p);
            let is_test = p.contains("/tests/") || p.contains("/__tests__/") || filename.contains(".test.") || filename.contains(".spec.");
            if is_test { js_tests += 1; } else { js_files += 1; }
        }
        if js_files > 10 && js_tests == 0 {
            return vec![Finding { id: "JS_TESTS_MISSING".into(), title: "Missing JavaScript/TypeScript tests".into(), description: format!("Detected {js_files} JS/TS source files but no corresponding test files."), path: None, severity: Severity::High, remediation: Some("Add unit tests (e.g., Jest, Vitest, or Mocha) under tests/ or __tests__/ with *.test.ts(x) naming.".into()), tags: vec!["tests".into(),"javascript".into(),"typescript".into()], kind: Some(FindingKind::Testing), score: None, created_at: Some(Utc::now().to_rfc3339()) }];
        }
        vec![]
    }
}

pub fn default_composite() -> CompositeAnalyzer {
    let cfg = AnalyzerConfig::from_env();
    let mut comp = CompositeAnalyzer::new()
        .register(ReadmePresence)
    .register(LicensePresence)
    .register(CiWorkflowPresence)
        .register(TestCoverageHeuristic)
    .register(JsTestHeuristic)
    .register(PythonTestHeuristic)
        .register(LargeFileDetector { threshold: cfg.large_file_threshold })
        .register(TodoCommentDetector)
        .register(ExtensionDiversity)
        .register(SecretScanner)
        .register(DependencyWildcard)
        .register(GitChurnHotspots { max_files: cfg.git_churn_max_files, min_commits: cfg.git_churn_min_commits });
    // Conditional analyzers (expensive / external tooling)
    if std::env::var("FKS_ENABLE_CLIPPY").ok().as_deref() == Some("1") { comp = comp.register(ClippyAnalyzer); }
    if std::env::var("FKS_ENABLE_AUDIT").ok().as_deref() == Some("1") { comp = comp.register(CargoAuditAnalyzer); }
    comp
}

// Convert findings into simplified task strings
pub fn findings_to_tasks(findings: &[Finding]) -> Vec<String> {
    findings.iter().map(|f| format!("[{:?}] {} - {}", f.severity, f.title, f.remediation.as_deref().unwrap_or("(no remediation)"))).collect()
}

// -------- Git Churn Analyzer ----------
pub struct GitChurnHotspots { pub max_files: usize, pub min_commits: usize }
impl Analyzer for GitChurnHotspots {
    fn name(&self) -> &'static str { "git_churn_hotspots" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        // Run a lightweight git log command; ignore if fails (e.g., not a git repo)
        let output = Command::new("git")
            .args(["log","--name-only","--pretty=format:%H","--since=30.days"])
            .output();
        let Ok(out) = output else { return vec![] };
        if !out.status.success() { return vec![]; }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for line in text.lines() { if line.is_empty() || line.len()==40 { continue; } // skip commit hash lines
            counts.entry(line).and_modify(|c| *c+=1).or_insert(1);
        }
        // Merge with existing scanned paths only
        let existing: HashMap<_,_> = scan.files.iter().map(|f| (f.path.as_str(), f)).collect();
        let mut items: Vec<(&str, usize)> = counts.into_iter().filter(|(p,c)| *c >= self.min_commits && existing.contains_key(*p)).collect();
        items.sort_by(|a,b| b.1.cmp(&a.1));
        items.truncate(self.max_files);
        items.into_iter().map(|(path, commits)| {
            let sev = if commits > 30 { Severity::High } else if commits > 15 { Severity::Medium } else { Severity::Low };
            Finding { id: format!("GIT_CHURN::{path}"), title: "High git churn".to_string(), description: format!("File {path} changed {commits} times in last 30 days"), path: Some(path.to_string()), severity: sev, remediation: Some("Stabilize this hotspot: add/expand tests, refactor to reduce churn.".to_string()), tags: vec!["git".into(),"hotspot".into()], kind: Some(FindingKind::Hotspot), score: None, created_at: Some(Utc::now().to_rfc3339()) }
        }).collect()
    }
}

// -------- Secret Scanner Analyzer ----------
pub struct SecretScanner;
impl Analyzer for SecretScanner {
    fn name(&self) -> &'static str { "secret_scanner" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        static ALLOW_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
            std::env::var("FKS_SECRET_ALLOWLIST").ok()
                .map(|s| s.split(',').filter_map(|p| Regex::new(p.trim()).ok()).collect())
                .unwrap_or_else(|| Vec::new())
        });
        let patterns = vec![
            (Regex::new(r"(?i)api[_-]?key[\s:=]+[A-Za-z0-9_-]{16,}").unwrap(), "Possible API key"),
            (Regex::new(r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-").unwrap(), "Private key material"),
        // JavaScript/TypeScript test heuristic analyzer
            (Regex::new(r"(?i)secret[_-]?key[\s:=]+[A-Za-z0-9/+]{16,}").unwrap(), "Secret key literal"),
        ];
        let mut findings = Vec::new();
        'file: for f in &scan.files { if let Some(snippet) = &f.snippet { for (re, title) in &patterns { if re.is_match(snippet) {
            // Allowlist check: if any allow pattern matches the snippet, skip
            if ALLOW_PATTERNS.iter().any(|a| a.is_match(snippet)) { continue 'file; }
            findings.push(Finding { id: format!("SECRET::{title}::{}", f.path), title: (*title).to_string(), description: format!("Potential secret in {}", f.path), path: Some(f.path.clone()), severity: Severity::High, remediation: Some("Remove secret, rotate credential, and use env/config vault.".to_string()), tags: vec!["security".into(),"secret".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) });
            break; } } } }
        findings
    }
}

// -------- Dependency Wildcard Analyzer (Rust) ----------
pub struct DependencyWildcard;
impl Analyzer for DependencyWildcard {
    fn name(&self) -> &'static str { "dependency_wildcard" }
    fn analyze(&self, scan: &ScanSummary) -> Vec<Finding> {
        // Look for Cargo.toml files with version = "*"
        let mut findings = Vec::new();
        for f in &scan.files { if f.path.ends_with("Cargo.toml") { if let Some(snippet) = &f.snippet { if snippet.contains("= \"*\"") {
            findings.push(Finding { id: format!("CARGO_WILDCARD::{}", f.path), title: "Wildcard dependency version".to_string(), description: format!("Wildcard '*' version in {} may cause unreproducible builds", f.path), path: Some(f.path.clone()), severity: Severity::Medium, remediation: Some("Pin to a compatible semver range (e.g., ^1.2) or exact version.".to_string()), tags: vec!["deps".into(),"rust".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) });
        } } } }
        findings
    }
}

// -------- Clippy Analyzer (optional) ----------
pub struct ClippyAnalyzer;
impl Analyzer for ClippyAnalyzer {
    fn name(&self) -> &'static str { "clippy" }
    fn analyze(&self, _scan: &ScanSummary) -> Vec<Finding> {
        // Run only if Cargo.toml present
        if !std::path::Path::new("Cargo.toml").exists() { return vec![]; }
        let output = std::process::Command::new("cargo")
            .args(["clippy","--message-format","json","--","-A","clippy::needless_return"]) // example allow
            .output();
        let Ok(out) = output else { return vec![] };
        if !out.status.success() && out.stdout.is_empty() { return vec![]; }
        let mut findings = Vec::new();
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if !(line.contains("compiler-message") && line.contains("clippy")) { continue; }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                let code = val.pointer("/message/code/code").and_then(|v| v.as_str()).unwrap_or("");
                if !code.starts_with("clippy::") { continue; }
                let level = val.pointer("/message/level").and_then(|v| v.as_str()).unwrap_or("warning");
                let message = val.pointer("/message/message").and_then(|v| v.as_str()).unwrap_or("clippy lint");
                let file = val.pointer("/message/spans/0/file_name").and_then(|v| v.as_str()).map(|s| s.to_string());
                let sev = match level { "error" => Severity::High, "warning" => Severity::Medium, _ => Severity::Low };
                findings.push(Finding { id: format!("CLIPPY::{code}"), title: format!("Clippy: {code}"), description: message.to_string(), path: file, severity: sev, remediation: Some("Review clippy suggestion and apply recommended fix.".to_string()), tags: vec!["clippy".into(),"lint".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) });
            }
        }
        findings
    }
}

// -------- Cargo Audit Analyzer (optional) ----------
pub struct CargoAuditAnalyzer;
impl Analyzer for CargoAuditAnalyzer {
    fn name(&self) -> &'static str { "cargo_audit" }
    fn analyze(&self, _scan: &ScanSummary) -> Vec<Finding> {
        if !std::path::Path::new("Cargo.lock").exists() { return vec![]; }
        let output = std::process::Command::new("cargo").args(["audit","--json"]).output();
        let Ok(out) = output else { return vec![] };
        if !out.status.success() && out.stdout.is_empty() { return vec![]; }
        let json: serde_json::Value = match serde_json::from_slice(&out.stdout) { Ok(v)=>v, Err(_)=> return vec![] };
        let mut findings = Vec::new();
        if let Some(list) = json.pointer("/vulnerabilities/list").and_then(|v| v.as_array()) {
            for vuln in list {
                let id = vuln.pointer("/advisory/id").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
                let package = vuln.pointer("/package/name").and_then(|v| v.as_str()).unwrap_or("?");
                let title = vuln.pointer("/advisory/title").and_then(|v| v.as_str()).unwrap_or("Vulnerability");
                let severity_s = vuln.pointer("/advisory/aliases/0").and_then(|_| vuln.pointer("/advisory/metadata/severity")).and_then(|v| v.as_str()).unwrap_or("medium");
                let patched = vuln.pointer("/advisory/versions/patched/0").and_then(|v| v.as_str()).unwrap_or("(update)");
                let severity = match severity_s.to_lowercase().as_str() { "critical" => Severity::Critical, "high" => Severity::High, "medium" => Severity::Medium, "low" => Severity::Low, _ => Severity::Low };
                let desc = format!("{title} in crate {package} (advisory {id})");
                let remediation = format!("Update {package} to {patched} or later (cargo update -p {package})");
                findings.push(Finding { id: format!("RUSTSEC::{id}"), title: format!("Vuln: {title}"), description: desc, path: None, severity, remediation: Some(remediation), tags: vec!["security".into(),"dependency".into()], kind: Some(FindingKind::Maintenance), score: None, created_at: Some(Utc::now().to_rfc3339()) });
            }
        }
        findings
    }
}

// Scoring logic: base severity weight + bonuses for certain kinds
fn score_findings(report: &mut FindingsReport) {
    // Configurable age decay via env or toml (lazy static to avoid repeated file IO)
    static AGE_CONF: Lazy<(f32,f32,f32)> = Lazy::new(|| {
        let mut start_days: f32 = 7.0;
        let mut end_days: f32 = 30.0;
        let mut min_factor: f32 = 0.5;
        if let Ok(cfg_text) = std::fs::read_to_string("fks_analyze.toml") {
            if let Ok(doc) = cfg_text.parse::<toml::Value>() {
                if let Some(v) = doc.get("age_decay_start_days").and_then(|v| v.as_integer()) { start_days = v as f32; }
                if let Some(v) = doc.get("age_decay_end_days").and_then(|v| v.as_integer()) { end_days = v as f32; }
                if let Some(v) = doc.get("age_decay_min_factor").and_then(|v| v.as_float()) { min_factor = v as f32; }
            }
        }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_START_DAYS") { if let Ok(v) = s.parse() { start_days = v; } }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_END_DAYS") { if let Ok(v) = s.parse() { end_days = v; } }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_MIN_FACTOR") { if let Ok(v) = s.parse() { min_factor = v; } }
        (start_days, end_days, min_factor)
    });
    let (start_days, end_days, min_factor) = *AGE_CONF;
    for f in &mut report.findings {
        let base = match f.severity { Severity::Info => 1.0, Severity::Low => 2.0, Severity::Medium => 4.0, Severity::High => 8.0, Severity::Critical => 13.0 };
        let bonus = match f.kind { Some(FindingKind::Hotspot) => 2.5, Some(FindingKind::Testing) => 3.0, Some(FindingKind::Documentation)|Some(FindingKind::DocsGap) => 1.0, _ => 0.0 };
        let decay = if let Some(ts) = &f.created_at { if let Ok(dt) = ts.parse::<DateTime<Utc>>() {
            let age_days = (Utc::now() - dt).num_seconds() as f32 / 86400.0;
            if age_days <= start_days { 1.0 } else if age_days >= end_days { min_factor } else {
                let span = (end_days - start_days).max(1.0);
                let prog = (age_days - start_days) / span;
                1.0 - prog * (1.0 - min_factor)
            }
        } else { 1.0 } } else { 1.0 };
        f.score = Some((base + bonus) * decay);
        if f.created_at.is_none() { f.created_at = Some(Utc::now().to_rfc3339()); }
    }
    report.findings.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
}

pub fn top_focus_tasks(report: &FindingsReport, limit: usize) -> Vec<String> {
    report.findings.iter().take(limit).map(|f| format!("{:.1} [{:?}] {} - {}", f.score.unwrap_or(0.0), f.severity, f.title, f.remediation.clone().unwrap_or_default())).collect()
}

// -------- Analyzer runtime configuration ---------
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    pub large_file_threshold: u64,
    pub git_churn_max_files: usize,
    pub git_churn_min_commits: usize,
    pub suppress_ids: Vec<String>,
    pub min_severity: Option<Severity>,
    pub age_decay_start_days: f32,
    pub age_decay_end_days: f32,
    pub age_decay_min_factor: f32,
}

impl AnalyzerConfig {
    pub fn from_env() -> Self {
    let mut large_file_threshold = std::env::var("FKS_LARGE_FILE_THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(200*1024);
        let git_churn_max_files = std::env::var("FKS_GIT_CHURN_MAX_FILES").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
        let git_churn_min_commits = std::env::var("FKS_GIT_CHURN_MIN_COMMITS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        // Load optional fks_analyze.toml at repo root
        let mut suppress_ids: Vec<String> = Vec::new();
        let mut min_severity: Option<Severity> = None;
        let mut age_decay_start_days: f32 = 7.0;
        let mut age_decay_end_days: f32 = 30.0;
        let mut age_decay_min_factor: f32 = 0.5;
        if let Ok(cfg_text) = std::fs::read_to_string("fks_analyze.toml") {
            if let Ok(doc) = cfg_text.parse::<toml::Value>() {
                if let Some(arr) = doc.get("suppress").and_then(|v| v.get("ids")).and_then(|v| v.as_array()) {
                    suppress_ids = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                }
                if let Some(th) = doc.get("large_file_threshold").and_then(|v| v.as_integer()) { large_file_threshold = th as u64; }
                if let Some(ms) = doc.get("min_severity").and_then(|v| v.as_str()) { min_severity = parse_severity(ms); }
                if let Some(v) = doc.get("age_decay_start_days").and_then(|v| v.as_integer()) { age_decay_start_days = v as f32; }
                if let Some(v) = doc.get("age_decay_end_days").and_then(|v| v.as_integer()) { age_decay_end_days = v as f32; }
                if let Some(v) = doc.get("age_decay_min_factor").and_then(|v| v.as_float()) { age_decay_min_factor = v as f32; }
            }
        }
    if let Ok(ms) = std::env::var("FKS_MIN_SEVERITY") { if let Some(sev) = parse_severity(&ms) { min_severity = Some(sev); } }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_START_DAYS") { if let Ok(v) = s.parse() { age_decay_start_days = v; } }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_END_DAYS") { if let Ok(v) = s.parse() { age_decay_end_days = v; } }
    if let Ok(s) = std::env::var("FKS_AGE_DECAY_MIN_FACTOR") { if let Ok(v) = s.parse() { age_decay_min_factor = v; } }
        Self { large_file_threshold, git_churn_max_files, git_churn_min_commits, suppress_ids, min_severity, age_decay_start_days, age_decay_end_days, age_decay_min_factor }
    }
}

pub fn parse_severity(s: &str) -> Option<Severity> {
    match s.to_ascii_lowercase().as_str() {
        "info" => Some(Severity::Info),
        "low" => Some(Severity::Low),
        "medium"|"med" => Some(Severity::Medium),
        "high" => Some(Severity::High),
        "critical"|"crit" => Some(Severity::Critical),
        _ => None,
    }
}

// Apply suppression rules (config + inline fks-ignore comments) to a findings report.
pub fn apply_suppressions(report: &mut FindingsReport, cfg: &AnalyzerConfig, root: &str) {
    let before = report.findings.len();
    if !cfg.suppress_ids.is_empty() { report.findings.retain(|f| !cfg.suppress_ids.iter().any(|id| id == &f.id)); }
    // Inline suppression: // fks-ignore:ID1,ID2   or  # fks-ignore:ID
    let inline_re = Regex::new(r"(?i)fks-ignore:([A-Za-z0-9_:\-*,]+)").unwrap();
    let mut cache: HashMap<String, Vec<String>> = HashMap::new();
    let mut is_suppressed_path = |path: &str, finding_id: &str| -> bool {
        if !cache.contains_key(path) {
            let full = std::path::Path::new(root).join(path);
            let text = std::fs::read_to_string(&full).unwrap_or_default();
            let mut ids: Vec<String> = Vec::new();
            for line in text.lines() { if let Some(cap) = inline_re.captures(line) { if let Some(m) = cap.get(1) { ids.extend(m.as_str().split(',').map(|s| s.trim().to_string())); } } }
            cache.insert(path.to_string(), ids);
        }
        if let Some(ids) = cache.get(path) {
            ids.iter().any(|p| p == "*" || finding_id.starts_with(p))
        } else { false }
    };
    report.findings.retain(|f| {
        if let Some(p) = &f.path { if is_suppressed_path(p, &f.id) { return false; } }
        true
    });
    // Recompute counts after pruning
    let mut counts: BTreeMap<Severity, usize> = BTreeMap::new();
    for f in &report.findings { *counts.entry(f.severity.clone()).or_insert(0) += 1; }
    report.counts = counts;
    let after = report.findings.len();
    let suppressed_now = before.saturating_sub(after);
    report.meta.insert("suppressed".into(), suppressed_now.to_string());
}

pub fn filter_min_severity(report: &mut FindingsReport, min: Severity) {
    let before = report.findings.len();
    report.findings.retain(|f| f.severity >= min);
    let mut counts: BTreeMap<Severity, usize> = BTreeMap::new();
    for f in &report.findings { *counts.entry(f.severity.clone()).or_insert(0) += 1; }
    report.counts = counts;
    let after = report.findings.len();
    report.meta.insert("min_severity".into(), format!("{:?}", min));
    report.meta.insert("filtered_by_severity".into(), (before.saturating_sub(after)).to_string());
}
