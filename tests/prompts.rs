use fks_analyze::prompts;

#[test]
fn catalog_not_empty() {
    let cat = prompts::catalog();
    assert!(!cat.prompts.is_empty(), "Prompt catalog should not be empty");
}

#[test]
fn categories_unique() {
    let mut cats = prompts::list_categories();
    let len = cats.len();
    cats.sort();
    cats.dedup();
    assert_eq!(len, cats.len(), "Categories should be unique");
}

#[test]
fn markdown_renders() {
    let md = prompts::to_markdown();
    assert!(md.contains("# Prompt Catalog"));
}

#[test]
fn fill_template_top_exts() {
    use fks_analyze::prompts::{FillContext, fill_template};
    let ctx = FillContext { repo_name: "fks_analyze", sibling_repos: vec!["fks_api".into()], top_exts: vec![("rs".into(), 120), ("toml".into(), 3)], total_files: 123, primary_lang: "Rust".into() };
    let out = fill_template("Repo [REPO_NAME]; Sibs [LIST_REPOS]; Exts [TOP_EXTS]", &ctx);
    assert!(out.contains("Repo fks_analyze"));
    assert!(out.contains("Sibs fks_api"));
    assert!(out.contains("rs:120"));
}

#[test]
fn fuzzy_find_basic() {
    let hits = prompts::fuzzy_find("repo analytics");
    assert!(!hits.is_empty(), "Expected fuzzy hits for 'repo analytics'");
    assert!(hits.iter().any(|p| p.name == "Repo Analytics"));
}

#[test]
fn best_match_prefers_exact() {
    if let Some(p) = prompts::best_match("Repo Analytics") { assert_eq!(p.name, "Repo Analytics"); } else { panic!("No best_match"); }
}

#[cfg(feature = "server")]
#[tokio::test]
async fn http_prompt_fill_integration() {
    // Spin up app router directly and query /prompt
    use axum::Router;
    use tower::ServiceExt; // for oneshot
    if std::env::var("CI").is_err() { /* still run locally; skip condition if needed */ }
    let tmp_root = std::env::current_dir().unwrap();
    let app: Router = fks_analyze::server::build_app(tmp_root.display().to_string());
    let resp = app
        .oneshot(axum::http::Request::builder()
            .uri("/prompt?name=Repo%20Analytics&fill=true")
            .body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body_bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert!(body_str.contains("Repo Analytics"));
    assert!(body_str.contains("Context:"));
    assert!(body_str.contains("TOP_EXTS")==false, "placeholder should be replaced");
}
