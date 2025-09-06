use crate::analyzer::Finding;
use anyhow::Result;
use std::fs;

// Simple auto-fix suggestion generator and optional in-place fixer (opt-in)
// Currently supports: replacing wildcard versions in Cargo.toml, inserting README.md stub, creating CI workflow,
// adding LICENSE (MIT) file, and basic test scaffolds for missing test findings.

pub enum AutoFixAction {
    ReplaceInFile { path: String, search: String, replace: String },
    WriteFile { path: String, content: String, skip_if_exists: bool },
}

pub struct AutoFixPlan { pub actions: Vec<AutoFixAction> }

pub fn plan_autofixes(findings: &[Finding]) -> AutoFixPlan {
    let mut actions = Vec::new();
    for f in findings {
        match f.id.as_str() {
            id if id.starts_with("CARGO_WILDCARD::") => {
                if let Some(path) = &f.path { actions.push(AutoFixAction::ReplaceInFile { path: path.clone(), search: "= \"*\"".into(), replace: "= \"^1.0\"".into() }); }
            }
            "README_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: "README.md".into(), content: default_readme_stub(), skip_if_exists: true });
            }
            "CI_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: ".github/workflows/ci.yml".into(), content: default_ci_stub(), skip_if_exists: true });
            }
            "LICENSE_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: "LICENSE".into(), content: default_license_stub(), skip_if_exists: true });
            }
            "RUST_TESTS_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: "tests/basic_smoke.rs".into(), content: rust_test_stub(), skip_if_exists: true });
            }
            "PY_TESTS_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: "tests/test_smoke.py".into(), content: python_test_stub(), skip_if_exists: true });
            }
            "JS_TESTS_MISSING" => {
                actions.push(AutoFixAction::WriteFile { path: "tests/smoke.test.ts".into(), content: js_test_stub(), skip_if_exists: true });
                actions.push(AutoFixAction::WriteFile { path: "package.json".into(), content: js_package_stub(), skip_if_exists: true });
            }
            _ => {}
        }
    }
    AutoFixPlan { actions }
}

pub fn apply_plan(root: &str, plan: &AutoFixPlan, dry_run: bool) -> Result<()> {
    for act in &plan.actions {
        match act {
            AutoFixAction::ReplaceInFile { path, search, replace } => {
                let full = std::path::Path::new(root).join(path);
                if !full.exists() { continue; }
                let data = fs::read_to_string(&full)?;
                if !data.contains(search) { continue; }
                let new_data = data.replace(search, replace);
                if dry_run { print_diff(&path, &data, &new_data); }
                if !dry_run { fs::write(&full, new_data)?; }
            }
            AutoFixAction::WriteFile { path, content, skip_if_exists } => {
                let full = std::path::Path::new(root).join(path);
                if *skip_if_exists && full.exists() { continue; }
                if dry_run { print_create(&path, content); }
                if let Some(parent) = full.parent() { fs::create_dir_all(parent)?; }
                if !dry_run { fs::write(&full, content)?; }
            }
        }
    }
    Ok(())
}

// Serializable representation for JSON output
#[derive(serde::Serialize)]
pub struct SerializableAction { pub kind: &'static str, pub path: String, pub details: serde_json::Value }
#[derive(serde::Serialize)]
pub struct SerializablePlan { pub actions: Vec<SerializableAction> }

pub fn serialize_plan(plan: &AutoFixPlan) -> SerializablePlan {
    let mut actions = Vec::new();
    for a in &plan.actions {
        match a {
            AutoFixAction::ReplaceInFile { path, search, replace } => actions.push(SerializableAction { kind: "replace", path: path.clone(), details: serde_json::json!({"search":search, "replace":replace}) }),
            AutoFixAction::WriteFile { path, skip_if_exists, .. } => actions.push(SerializableAction { kind: "write", path: path.clone(), details: serde_json::json!({"skip_if_exists":skip_if_exists}) }),
        }
    }
    SerializablePlan { actions }
}

pub fn plan_unified_diff(root: &str, plan: &AutoFixPlan) -> Result<String> {
    let mut out = String::new();
    for a in &plan.actions {
        match a {
            AutoFixAction::ReplaceInFile { path, search, replace } => {
                let full = std::path::Path::new(root).join(path);
                if !full.exists() { continue; }
                let data = std::fs::read_to_string(&full)?;
                if !data.contains(search) { continue; }
                let new_data = data.replace(search, replace);
                out.push_str(&format!("diff --git a/{0} b/{0}\n--- a/{0}\n+++ b/{0}\n", path));
                // naive line diff
                for diff in diff_lines(&data, &new_data) { out.push_str(&diff); }
                out.push('\n');
            }
            AutoFixAction::WriteFile { path, content, skip_if_exists } => {
                let full = std::path::Path::new(root).join(path);
                if *skip_if_exists && full.exists() { continue; }
                out.push_str(&format!("diff --git a/{0} b/{0}\nnew file mode 100644\n--- /dev/null\n+++ b/{0}\n", path));
                for line in content.lines() { out.push_str(&format!("+{}\n", line)); }
                out.push('\n');
            }
        }
    }
    Ok(out)
}

fn diff_lines(old: &str, new: &str) -> Vec<String> {
    let o: Vec<&str> = old.lines().collect();
    let n: Vec<&str> = new.lines().collect();
    // Very small LCS-free heuristic: show full removal/add; acceptable for simple replacements
    if o.len() + n.len() > 5000 { // large file fallback
        let mut v = Vec::new();
        for l in &o { v.push(format!("-{}\n", l)); }
        for l in &n { v.push(format!("+{}\n", l)); }
        return v;
    }
    let mut v = Vec::new();
    for l in &o { if !n.contains(l) { v.push(format!("-{}\n", l)); } }
    for l in &n { if !o.contains(l) { v.push(format!("+{}\n", l)); } }
    v
}

fn print_diff(path: &str, old: &str, new: &str) {
    use similar::{TextDiff, ChangeTag};
    let diff = TextDiff::from_lines(old, new);
    println!("--- {} (planned)\n+++ {} (new)\n", path, path);
    for op in diff.ops() { for change in diff.iter_changes(op) { match change.tag() { ChangeTag::Delete => print!("-{}", change), ChangeTag::Insert => print!("+{}", change), ChangeTag::Equal => print!(" {}", change) }; } }
    println!();
}

fn print_create(path: &str, content: &str) {
    println!("++ {} (create)\n{}", path, content);
}

fn default_readme_stub() -> String {
"# Project Title\n\nProvide a concise description, setup instructions, and contribution guidelines.\n".into()
}

fn default_ci_stub() -> String {
"name: CI\n\non: [push, pull_request]\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions-rs/toolchain@v1\n        with:\n          toolchain: stable\n      - name: Build\n        run: cargo build --verbose\n      - name: Test\n        run: cargo test --all --quiet\n".into()
}

fn default_license_stub() -> String {
"MIT License\n\nCopyright (c) YEAR YOUR_NAME\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n".into()
}

fn rust_test_stub() -> String {
"#[test]\nfn smoke() {\n    assert_eq!(2+2, 4);\n}\n".into()
}

fn python_test_stub() -> String {
"def test_smoke():\n    assert 2 + 2 == 4\n".into()
}

fn js_test_stub() -> String {
"test('smoke', () => {\n  expect(2+2).toBe(4);\n});\n".into()
}

fn js_package_stub() -> String {
"{\n  \"name\": \"fks-js-tests\",\n  \"private\": true,\n  \"devDependencies\": {\n    \"vitest\": \"^1.0.0\"\n  },\n  \"scripts\": {\n    \"test\": \"vitest run\"\n  }\n}\n".into()
}
