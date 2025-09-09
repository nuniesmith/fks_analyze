/// Helpers for mapping friendly analyze kind aliases to legacy script numeric choices.
/// Centralizing logic to avoid duplication between CLI and Discord bot.

pub const ANALYZE_KIND_HELP: &str = "Analysis kinds:\n  all -> 1 (all languages)\n  python -> 2\n  rust -> 3\n  csharp (cs) -> 4\n  jsts (js|ts|node|react) -> 5\n  mdtxt (md|markdown|txt) -> 6\n  shbash (shell|sh) -> 7\n  docker (compose) -> 8";

/// Map a user supplied alias or numeric choice to the script choice string.
/// Returns the mapped numeric choice; if the input already looks like a number we pass it through.
pub fn map_kind<S: AsRef<str>>(input: S) -> String {
    let k = input.as_ref().to_lowercase();
    match k.as_str() {
        "all" => "1".into(),
        "python" => "2".into(),
        "rust" => "3".into(),
    "csharp" | "cs" => "4".into(),
    "jsts" | "js" | "ts" | "node" | "react" => "5".into(),
    "mdtxt" | "md" | "markdown" | "txt" => "6".into(),
    "shbash" | "shell" | "sh" => "7".into(),
    "docker" | "compose" => "8".into(),
        // If already numeric or unknown, return as-is (script may prompt or error).
        _ => k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_map_kind_aliases() {
        assert_eq!(map_kind("all"), "1");
        assert_eq!(map_kind("PYTHON"), "2");
        assert_eq!(map_kind("rust"), "3");
    assert_eq!(map_kind("csharp"), "4");
    assert_eq!(map_kind("cs"), "4");
    for a in ["jsts","js","ts","node","react"] { assert_eq!(map_kind(a), "5"); }
    for a in ["mdtxt","md","markdown","txt"] { assert_eq!(map_kind(a), "6"); }
    for a in ["shbash","shell","sh"] { assert_eq!(map_kind(a), "7"); }
    for a in ["docker","compose"] { assert_eq!(map_kind(a), "8"); }
        // Pass through numbers / unknown
        assert_eq!(map_kind("42"), "42");
        assert_eq!(map_kind("unknown"), "unknown");
    }
}
