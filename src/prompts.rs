use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub category: String,
    pub name: String,
    pub template: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCatalog { pub guidelines: String, pub prompts: Vec<PromptTemplate> }

// Static (compile-time) catalog built from docs/prompts.md content distilled into structured records.
// NOTE: Keep this in sync with docs/prompts.md. Only the core fenced code blocks are embedded here.
pub fn catalog() -> &'static PromptCatalog { &*CATALOG }

static GUIDELINES: &str = "Context Inclusion: prepend repo summary (up to 32). DRY Focus: prefer shared_ repos. Tech Stack: Docker/Compose -> future K8s; langs: Python, Rust, C#, JS/TS/React, HTML/CSS, Shell, Jupyter, Java, Batchfile. Mention scale for broad tasks.";

static CATALOG: once_cell::sync::Lazy<PromptCatalog> = once_cell::sync::Lazy::new(|| {
    let mut v: Vec<PromptTemplate> = Vec::new();
    macro_rules! p { ($cat:expr,$name:expr,$tmpl:expr,$desc:expr) => { v.push(PromptTemplate{category:$cat.into(), name:$name.into(), template:$tmpl.trim().into(), description:$desc.into()}); } }
    // 1 Code Review & Refactoring
    p!("Code Review","General Code Review", r#"Review the following code from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Suggest improvements for performance, readability, and DRY principles. If applicable, recommend moving shared logic to a shared repo like shared_[LANGUAGE] or shared_scripts. Consider our stack: Docker/Docker Compose, etc."#, "General multi-language code review");
    p!("Code Review","DRY-Focused Refactoring", r#"Analyze this code snippet from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Identify duplicated patterns and suggest refactoring to make it DRY. Propose extracting common parts to a shared repo (e.g., shared_python for Python utils, shared_rust for Rust crates). Ensure compatibility with Docker setups in shared_docker."#, "Identify duplication & refactor");
    p!("Code Review","Cross-Repo Consistency", r#"Compare code patterns across these repos: [LIST_REPOS]. Look for inconsistencies in [SPECIFIC_ASPECT]. Suggest standardizations to promote DRY, using shared repos where possible."#, "Cross repo pattern alignment");
    p!("Code Review","API Endpoint Review", r#"Review this API code from [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Check for REST/GraphQL best practices, security (auth), and performance. Suggest DRY integrations with shared_schema or shared_python."#, "API quality & security");
    p!("Code Review","Dependency Audit", r#"Audit dependencies in [REPO_NAME] ([LANGUAGE]). List outdated/vulnerable packages and suggest updates. Propose centralizing common deps in a shared repo like shared_python or shared_rust for DRY management across 32 repos."#, "Dependency freshness & risk");
    // 2 Docker & Containerization
    p!("Docker","Dockerfile Optimization", r#"Optimize this Dockerfile from [REPO_NAME]: [PASTE_DOCKERFILE_HERE]. Focus on multi-stage builds, security, and efficiency. Integrate best practices from shared_docker. Prepare for future K8s migration."#, "Improve image build & size");
    p!("Docker","Compose Setup", r#"Help set up a docker-compose.yml for [REPO_NAME] integrating services from [RELATED_REPOS]. Include volumes, networks, env vars. Reuse snippets via shared_docker & shared_scripts. Ensure scalability for K8s."#, "Compose orchestration");
    p!("Docker","Migration to Kubernetes", r#"Outline steps to migrate this Docker Compose setup from [REPO_NAME] to Kubernetes: [PASTE_COMPOSE_YML_HERE]. Suggest Helm charts or manifests, leveraging shared_docker shared_nginx."#, "K8s readiness plan");
    p!("Docker","Multi-Repo Orchestration", r#"Create a top-level Docker Compose or K8s config to orchestrate [LIST_REPOS]. Use shared_docker for base images and shared_scripts for entrypoints. Focus on DRY for new service rollout."#, "Fleet orchestration");
    p!("Docker","Container Security Scan", r#"Suggest tools and scripts to scan Docker images in [REPO_NAME]. Integrate with CI (shared_actions). Prepare for K8s security policies."#, "Image security scanning");
    // 3 Language Specific (subset for brevity - others can be added as needed)
    p!("Language","Rust Feature", r#"Implement [FEATURE_DESCRIPTION] in Rust for [REPO_NAME]. Emphasize safety, performance, and error handling. Suggest crates from shared_rust. Build & test with Docker shared_scripts."#, "Rust feature implementation");
    p!("Language","Python Task", r#"Write or refactor Python code for [TASK_DESCRIPTION] in [REPO_NAME]. Use type hints & async where helpful. Integrate utilities from shared_python. Test via Docker (shared_docker)."#, "Python work refactoring");
    p!("Language","Node/React Component", r#"Create or improve [COMPONENT/PAGE_DESCRIPTION] in [REPO_NAME] using [JS/TS/React]. Reuse components from shared_react. Bundle with Docker & nginx via shared_nginx."#, "Frontend component");
    p!("Language","Shell Script", r#"Write a shell script for [TASK_DESCRIPTION] in [REPO_NAME]. Make it idempotent & portable. Source common helpers from shared_scripts. Integrate Docker commands."#, "Shell automation");
    // 4 Repo Management
    p!("Repo Mgmt","New Repo Setup", r#"Guide me on setting up a new repo named [NEW_REPO_NAME] for [PURPOSE]. Include structure, MIT LICENSE, integrations with shared_docker & shared_scripts. Optimize for DRY across 32 repos."#, "Bootstrap new repository");
    p!("Repo Mgmt","Shared Repo Integration", r#"Suggest how to integrate [SHARED_REPO] into [TARGET_REPO]. Provide import/examples. Ensure DRY across ecosystem."#, "Adopt shared utilities");
    p!("Repo Mgmt","Repo Consolidation", r#"Review these repos: [LIST_REPOS]. Identify overlapping code/features; propose moving to shared repo (e.g., shared_schema). Provide migration steps."#, "Reduce duplication via consolidation");
    p!("Repo Mgmt","CI/CD Pipeline", r#"Set up GitHub Actions for [REPO_NAME] using shared_actions: build images, run tests, push artifacts. Reusable for new repos."#, "Automate build & test");
    p!("Repo Mgmt","Project Template", r#"Create a template (cookiecutter) for new [TYPE] repos using shared_docker, shared_scripts, MIT license, standard CI."#, "Template generation");
    // 5 Debugging
    p!("Debug","Debug Code Issue", r#"Debug this error in [REPO_NAME] ([LANGUAGE]): [ERROR_MESSAGE]. Consider Docker env from shared_docker. Suggest fixes and preventive DRY improvements."#, "Root cause analysis");
    p!("Debug","Performance Optimization", r#"Optimize performance for [FEATURE_DESCRIPTION] in [REPO_NAME]. Profile, identify hotspots, propose improvements leveraging shared tools."#, "Perf tuning");
    p!("Debug","Logging Standardization", r#"Review logging in [REPO_NAME] and suggest standardization (structured, levels). Integrate shared logging helpers (shared_python/shared_rust)."#, "Consistent logging");
    // 6 Testing & QA
    p!("Testing","Unit Test Generation", r#"Generate unit tests for this code in [REPO_NAME] ([LANGUAGE]): [PASTE_CODE_HERE]. Use appropriate framework. Suggest shared test utils repo."#, "Add unit tests");
    p!("Testing","Integration Test Setup", r#"Set up integration tests for [REPO_NAME] involving [RELATED_REPOS]. Use docker-compose + shared_scripts for orchestration."#, "Cross-service tests");
    p!("Testing","Coverage Analysis", r#"Analyze test coverage for [REPO_NAME]; identify critical gaps; propose test additions with DRY shared patterns."#, "Improve coverage");
    // 7 Security & Compliance
    p!("Security","Security Audit", r#"Perform a high-level security review of [REPO_NAME] focusing on [ASPECT]. Suggest fixes referencing shared crypto/auth utilities."#, "Security review");
    p!("Security","Vulnerability Scan", r#"Recommend tools & steps to scan [REPO_NAME] for vulnerabilities; integrate into CI (shared_actions) & Docker image scans."#, "Vulnerability tooling");
    // 8 Scaling & Deployment
    p!("Scaling","Scaling Strategy", r#"Outline scaling for [REPO_NAME] from Docker Compose to K8s. Use shared_docker base and shared_nginx for ingress."#, "Growth plan");
    p!("Scaling","Deployment Pipeline", r#"Design a deployment pipeline for [LIST_REPOS]. Include build, test, security scan, deploy, rollback; reuse shared_scripts."#, "Deployment workflow");
    p!("Scaling","Monitoring Setup", r#"Suggest monitoring/metrics stack for [REPO_NAME] (Prometheus/Grafana etc). Externalize reusable configs to shared repo."#, "Observability setup");
    // 9 Collaboration
    p!("Collaboration","Branching Strategy", r#"Recommend a git branching model for [REPO_NAME] considering shared repos. Compare trunk vs GitFlow with rationale."#, "Version control model");
    p!("Collaboration","PR Template", r#"Create a pull request template for [REPO_NAME] emphasizing DRY checks, tests, docs links."#, "Improve PR quality");
    p!("Collaboration","Contributor Guidelines", r#"Generate CONTRIBUTING.md for [REPO_NAME] covering stack, DRY principles, license, code style."#, "Onboarding doc");
    // 10 Misc
    p!("Misc","Documentation Generation", r#"Generate README.md or docs for [REPO_NAME], covering setup with Docker Compose, languages, links to shared repos."#, "Author docs");
    p!("Misc","Feature Brainstorm", r#"Brainstorm ideas for [FEATURE_TYPE] across repos: [LIST_REPOS]. Prioritize DRY leveraging shared_rust/shared_python."#, "Idea generation");
    p!("Misc","Repo Analytics", r#"Suggest ways to analyze activity across [LIST_REPOS] (update times, commit frequency). Provide shell script patterns using shared_scripts. Observed top file extensions in current repo: [TOP_EXTS]."#, "Activity insights with extension context");

    PromptCatalog { guidelines: GUIDELINES.into(), prompts: v }
});

pub fn list_categories() -> Vec<String> {
    let mut cats: Vec<String> = catalog().prompts.iter().map(|p| p.category.clone()).collect();
    cats.sort(); cats.dedup(); cats
}

pub fn by_category(cat: &str) -> Vec<&'static PromptTemplate> {
    catalog().prompts.iter().filter(|p| p.category.eq_ignore_ascii_case(cat)).collect()
}

pub fn find(name: &str) -> Option<&'static PromptTemplate> {
    catalog().prompts.iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

/// Fuzzy search for prompts by name. Heuristic scoring:
/// 0 = exact (case insensitive)
/// 1 = starts with (case insensitive)
/// 2 = contains (case insensitive)
/// 3 = subsequence (characters appear in order)
/// Returns matches sorted by score then name length.
#[allow(dead_code)]
pub fn fuzzy_find(query: &str) -> Vec<&'static PromptTemplate> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() { return Vec::new(); }
    let mut scored: Vec<(u8, usize, &'static PromptTemplate)> = Vec::new();
    for p in &catalog().prompts {
        let name_l = p.name.to_ascii_lowercase();
        let score_opt = if name_l == q { Some(0) }
            else if name_l.starts_with(&q) { Some(1) }
            else if name_l.contains(&q) { Some(2) }
            else if is_subsequence(&q, &name_l) { Some(3) }
            else { None };
        if let Some(score) = score_opt { scored.push((score, p.name.len(), p)); }
    }
    scored.sort_by(|a,b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1))); // prefer better score then shorter name
    scored.into_iter().map(|(_,_,p)| p).collect()
}

/// Return best match falling back from exact to other fuzzy modes.
#[allow(dead_code)]
pub fn best_match(query: &str) -> Option<&'static PromptTemplate> {
    // Fast path exact
    if let Some(p) = find(query) { return Some(p); }
    let matches = fuzzy_find(query);
    // If multiple exact-level results would have been caught above.
    matches.first().cloned()
}

#[allow(dead_code)]
fn is_subsequence(needle: &str, hay: &str) -> bool {
    if needle.len() > hay.len() { return false; }
    let mut it = hay.chars();
    for ch in needle.chars() {
        if let Some(_) = it.by_ref().find(|c| c == &ch) { continue; } else { return false; }
    }
    true
}

pub fn to_markdown() -> String {
    let mut out = String::new();
    out.push_str("# Prompt Catalog\n\n");
    out.push_str("Guidelines: "); out.push_str(GUIDELINES); out.push_str("\n\n");
    // Use list_categories + by_category so by_category is exercised and not considered dead code.
    for cat in list_categories() {
        out.push_str(&format!("## {}\n\n", cat));
        for pt in by_category(&cat) {
            // Exercise find() so it is not considered dead code.
            let _ = find(&pt.name);
            out.push_str(&format!("### {}\n{}\n\n```\n{}\n```\n\n", pt.name, pt.description, pt.template));
        }
    }
    out
}

/// Lightweight context used for filling prompt placeholders.
pub struct FillContext<'a> {
    pub repo_name: &'a str,
    pub sibling_repos: Vec<String>,
    pub top_exts: Vec<(String, usize)>,
    pub total_files: usize,
    pub primary_lang: String,
}

/// Fill supported placeholders in a template.
/// Currently replaces:
/// - [REPO_NAME]
/// - [LIST_REPOS]
/// Leaves any other bracketed tokens for user refinement.
pub fn fill_template(tpl: &str, ctx: &FillContext) -> String {
    let mut out = tpl.to_string();
    out = out.replace("[REPO_NAME]", ctx.repo_name);
    if out.contains("[LIST_REPOS]") {
        let list = if ctx.sibling_repos.is_empty() { ctx.repo_name.to_string() } else { ctx.sibling_repos.join(", ") };
        out = out.replace("[LIST_REPOS]", &list);
    }
    if out.contains("[TOP_EXTS]") {
        let top = if ctx.top_exts.is_empty() { "(no-ext-data)".to_string() } else { ctx.top_exts.iter().take(8).map(|(e,c)| format!("{e}:{c}")).collect::<Vec<_>>().join(", ") };
        out = out.replace("[TOP_EXTS]", &top);
    }
    if out.contains("[TOTAL_FILES]") {
        out = out.replace("[TOTAL_FILES]", &ctx.total_files.to_string());
    }
    if out.contains("[PRIMARY_LANG]") {
        out = out.replace("[PRIMARY_LANG]", &ctx.primary_lang);
    }
    out
}

use std::time::{Instant, Duration};
use once_cell::sync::Lazy;
use std::sync::RwLock;

#[allow(dead_code)]
struct CachedContext {
    built_at: Instant,
    repo_root: String,
    ctx: FillContext<'static>,
}

#[allow(dead_code)]
static CTX_CACHE: Lazy<RwLock<Option<CachedContext>>> = Lazy::new(|| RwLock::new(None));

/// Build (or fetch cached) FillContext for a repo root. Caches for TTL to avoid repeated scans.
#[allow(dead_code)]
pub fn build_fill_context(root: &str, ttl: Duration) -> FillContext<'static> {
    // Check cache first
    if let Ok(r) = CTX_CACHE.read() {
    if let Some(c) = &*r { if c.repo_root == root && c.built_at.elapsed() < ttl { return FillContext { repo_name: c.ctx.repo_name, sibling_repos: c.ctx.sibling_repos.clone(), top_exts: c.ctx.top_exts.clone(), total_files: c.ctx.total_files, primary_lang: c.ctx.primary_lang.clone() }; } }
    }
    // Compute new
    let repo_name_owned = std::path::Path::new(root).file_name().and_then(|s| s.to_str()).unwrap_or("repo").to_string();
    let parent = std::path::Path::new(root).parent().unwrap_or(std::path::Path::new(root));
    let siblings: Vec<String> = std::fs::read_dir(parent).ok().map(|rd| rd.filter_map(|e| e.ok()).filter_map(|e| {
        let n = e.file_name().to_string_lossy().to_string(); if n.starts_with("fks_") && e.path().is_dir() { Some(n) } else { None }
    }).collect()).unwrap_or_default();
    // Quick & cheap top extensions by walking one level deep (fallback: empty)
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if let Ok(read) = walkdir::WalkDir::new(root).max_depth(3).into_iter().collect::<Result<Vec<_>,_>>() { for entry in read { if entry.file_type().is_file() { if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) { *counts.entry(ext.to_string()).or_insert(0) += 1; } } } }
    let mut top_exts: Vec<(String, usize)> = counts.into_iter().collect();
    top_exts.sort_by(|a,b| b.1.cmp(&a.1));
    top_exts.truncate(12);
    let total_files: usize = top_exts.iter().map(|(_, c)| *c).sum();
    // Map primary extension to language label
    let primary_lang = top_exts.first().map(|(ext, _)| match ext.as_str() {
        "rs" => "Rust",
        "py" => "Python",
        "ts" => "TypeScript",
        "js" => "JavaScript",
        "tsx" => "TypeScript/React",
        "jsx" => "JavaScript/React",
        "sh" => "Shell",
        "yml" | "yaml" => "YAML",
        "toml" => "TOML",
        "md" => "Markdown",
        "cs" => "C#",
        "java" => "Java",
        "html" => "HTML",
        "css" => "CSS",
        "json" => "JSON",
        other => {
            // Fallback: uppercase extension
            Box::leak(other.to_uppercase().into_boxed_str())
        }
    }.to_string()).unwrap_or_else(|| "Unknown".to_string());
    // Leak strings to create 'static references (safe for process lifetime) to store in cache context
    let repo_static: &'static str = Box::leak(repo_name_owned.clone().into_boxed_str());
    let ctx_new = FillContext { repo_name: repo_static, sibling_repos: siblings.clone(), top_exts: top_exts.clone(), total_files, primary_lang: primary_lang.clone() };
    if let Ok(mut w) = CTX_CACHE.write() {
        *w = Some(CachedContext { built_at: Instant::now(), repo_root: root.to_string(), ctx: FillContext { repo_name: repo_static, sibling_repos: siblings, top_exts: top_exts, total_files, primary_lang } });
    }
    ctx_new
}
