#[cfg(feature = "server")]
use fks_analyze::{scan, persist};
#[cfg(feature = "server")]
use fks_analyze::server::build_app;
#[cfg(feature = "server")]
use axum::http::StatusCode;
#[cfg(feature = "server")]
use tower::ServiceExt; // for oneshot

#[cfg(feature = "server")]
#[tokio::test]
async fn health_works() {
    let app = build_app(std::env::current_dir().unwrap().display().to_string());
    let response = app
        .oneshot(axum::http::Request::builder().uri("/health").body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn search_and_diff_endpoints() {
    use std::fs;
    use std::time::Duration;
    use tokio::time::sleep;
    let root = tempfile::tempdir().unwrap();
    // create files
    fs::write(root.path().join("a.txt"), "hello executor world").unwrap();
    // build index by calling search (open_or_build will create)
    let app = build_app(root.path().display().to_string());
    let resp = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/search?query=executor&limit=10")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // create second snapshot for diff
    // First snapshot
    {
        let scan = scan::scan_repo(root.path().to_str().unwrap(), 0).unwrap();
        persist::save_snapshot(root.path().to_str().unwrap(), &scan).unwrap();
    }
    sleep(Duration::from_millis(1100)).await; // ensure timestamp difference
    fs::write(root.path().join("b.txt"), "second file").unwrap();
    {
        let scan = scan::scan_repo(root.path().to_str().unwrap(), 0).unwrap();
        persist::save_snapshot(root.path().to_str().unwrap(), &scan).unwrap();
    }
    let app2 = build_app(root.path().display().to_string());
    let diff_resp = app2
        .oneshot(
            axum::http::Request::builder()
                .uri("/diff?format=json")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(diff_resp.status(), StatusCode::OK);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn index_endpoint_rebuilds() {
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("a.rs"), "fn main(){} // token").unwrap();
    let app = build_app(root.path().display().to_string());
    let resp = app
        .oneshot(axum::http::Request::builder().uri("/index").body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn version_endpoint() {
    let app = build_app(std::env::current_dir().unwrap().display().to_string());
    let resp = app.oneshot(axum::http::Request::builder().uri("/version").body(axum::body::Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[cfg(feature = "server")]
#[tokio::test]
async fn stats_endpoint() {
    let app = build_app(std::env::current_dir().unwrap().display().to_string());
    let resp = app.oneshot(axum::http::Request::builder().uri("/stats").body(axum::body::Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
