# fks_analyze

Rust service and tooling to scan, diff, suggest tasks, and interact via Discord for the FKS mono‑environment.

## Features

- Repo / multi-repo scanning (`scan`, `scan-all`)
- Snapshot persistence & diffs (`scan --save`, `diff`)
- Task suggestion engine (`suggest`, optional LLM enrich with Ollama)
- File system watch mode (`watch`) producing live diffs
- Lightweight substring search index (`index`, `search`)
- Optional Discord bot (feature flag `discord`)
- HTTP server with `/health`, `/version`, `/stats`, `/scan`, `/suggest`, `/diff`, `/search`, `/index` (feature flag `server`, enabled by default)
- Structured AI prompt catalog (CLI: `prompts` / `prompt`, HTTP: `/prompts` / `/prompt`) with optional context fill (repo name, sibling repos, top extensions)
- Shell-driven multi-language code analysis (`analyze` command & Discord /analyze) integrated (previously legacy)
- Rich analyzer framework with configurable suppression, minimum severity filtering, and tunable age-based scoring decay

## CLI Quick Start

```bash
# Build
cargo build

# Multi-repo scan example
cargo run -- scan-all --include-shared_repos > multi.json

# Build index & search
cargo run -- index --content-bytes 2048

# Common commands
cargo run -- scan --save
cargo run -- scan --save --compress true --retention 5 --retention-days 14
cargo run -- prune --keep 10 --max-age-days 30
cargo run -- prune --keep 5 --dry-run
cargo run -- stats --format json --limit 5
cargo run -- diff
cargo run -- diff --unified --unified-context 5 --unified-max 50000 --format json | jq '.'
cargo run -- report --min-severity high --kinds Testing,Hotspot --format json | jq '.'
cargo run -- suggest --enrich
cargo run -- prompts --format json | jq '.categories'
cargo run -- prompts --category Docker --format text
cargo run -- prompt --name "Repo Analytics" --fill --format json | jq '.filled'
cargo run -- search --query executor --limit 10
env RUST_LOG=info cargo run -- watch --content-bytes 1024
cargo run -- findings --fail-on medium            # exit 2 if any Medium+
cargo run -- findings --fail-on high --pretty > findings.json  # pretty JSON + gating
cargo run -- report --fail-on high --format json  # gate on high severity using report summary
cargo run -- report --fail-on-delta medium --format html > report.html  # fail if medium count increased & generate HTML
cargo run -- pipeline --format json --content-bytes 512 > pipeline.json  # structured pipeline output

# Ignore patterns (comma separated globs)
cargo run -- scan --ignore "**/target/**,**/*.log" --content-bytes 256
cargo run -- scan-all --ignore "**/target/**" --include-shared_repos
cargo run -- scan-all --save --compress true --retention 3 --content-bytes 512

# HTTP server examples (after starting with `cargo run -- serve --port 4802`)
curl -s 'http://localhost:4802/version'
curl -s 'http://localhost:4802/health'
curl -s 'http://localhost:4802/stats' | jq '.'
curl -s 'http://localhost:4802/scan?ignore=**/target/**&content_bytes=512' | jq '.total_files'
curl -s 'http://localhost:4802/suggest?enrich=1' | head
curl -s 'http://localhost:4802/prompts?format=json&category=Security' | jq '.count'
curl -s 'http://localhost:4802/prompts?format=md' | head
curl -s 'http://localhost:4802/prompt?name=Repo%20Analytics&fill=1' | sed -n '1,30p'
curl -s 'http://localhost:4802/diff?format=text'
curl -s 'http://localhost:4802/index'
curl -s 'http://localhost:4802/search?query=executor&limit=5' | jq '.'
```

### CI Severity Gating

Use the `findings` subcommand with `--fail-on <severity>` inside your CI workflow to break the build when issues at or above a threshold appear. The process exits with code 2 when triggered (0 otherwise).

Examples (GitHub Actions steps):

```bash
run: |
   cargo run -- findings --fail-on medium --pretty > findings.json
```

```bash
run: |
   cargo run -- report --fail-on high --format json > report.json
```

```bash
run: |
   cargo run -- pipeline --fail-on medium --content-bytes 0
```

Accepted severities: `info`, `low`, `medium`, `high`, `critical`.

Exit codes:

- 0: success (threshold not met)
- 2: fail-on threshold triggered (findings/report/pipeline)

Delta gating (`--fail-on-delta <severity>`) exits 2 only when the count for that exact severity increases versus the previous saved findings snapshot (requires at least two snapshots). This allows enforcing "no regressions" while permitting existing debt.

Examples:


```bash
cargo run -- findings --save --fail-on-delta high        # fail only if new high issues appeared
cargo run -- report --fail-on-delta medium --format html > report.html
cargo run -- pipeline --save-findings --fail-on-delta high  # pipeline regression gating
```


If fewer than two findings snapshots exist, delta gating is a no-op (exits 0).

You can gate different stages (raw findings, aggregated report, or full pipeline) depending on your workflow maturity.


## Feature Flags

Default features: `server`

Optional:

- `discord` – enable Discord bot commands
- `full` – convenience aggregate = server + discord

Examples:

```bash
cargo run --no-default-features --features "server,discord" -- serve --port 4802
cargo run --features full -- bot
```

## Discord Bot Setup

Requires building with feature:

```bash
cargo run --no-default-features --features "discord" -- bot
```

### 1. Create Application & Bot (Discord Developer Portal)

1. Visit <https://discord.com/developers/applications>
2. New Application -> name (e.g., FKS Analyze Bot)
3. Note Application (Client) ID and Public Key.
4. Bot tab -> Add Bot -> Confirm.
5. Reset Token and copy (put into `.env` as `DISCORD_TOKEN`).
6. Under Privileged Gateway Intents enable:
   - MESSAGE CONTENT INTENT (needed for prefix commands)

### 2. Permissions / Intents

Minimal recommended bot scopes & permissions:

- Scopes: `bot` (later `applications.commands` for slash)
- Bot Permissions: Send Messages, Read Message History.

Permission integer (basic: 2048 for Send Messages + 1024 Read History = 3072). Use the portal permission calculator for additional needs.

### 3. Generate OAuth2 Invite URL

Go to OAuth2 -> URL Generator:

- Scopes: `bot`
- Bot Permissions: select at least Send Messages, Read Message History.

Copy the generated URL and open it; pick the target server where you have permission to add bots.

(When slash commands added later: include `applications.commands` scope.)

### 4. Environment Variables

Copy `.env.example` to `.env` and fill values:

```bash
cp .env.example .env
```
 
Key vars:

- `DISCORD_TOKEN` (bot token)
- `DISCORD_GUILD_ALLOWLIST` (comma separated guild IDs) or legacy `DISCORD_GUILD_ID`
- `DISCORD_CHANNEL_ALLOWLIST` (comma separated) or legacy `DISCORD_CHANNEL_ID`
- `FKS_ALLOW_CONTENT_BYTES` (max file bytes returned via !file)
- `FKS_MAX_DISCORD_CHARS` (truncate long replies)
- `FKS_SNAPSHOT_FULL=1` (persist full file contents up to `FKS_SNAPSHOT_FULL_MAX` for richer diffs)
- `FKS_SNAPSHOT_FULL_MAX` (bytes cap for full content capture; default 200000)
- `FKS_SNAPSHOT_FULL_SELECTIVE=1` (capture full content only for changed files; overrides snippet limit)
- `FKS_SNAPSHOT_COMPRESS=1` (gzip snapshot JSON as .json.gz)
- `FKS_SNAPSHOT_RETENTION` (keep only latest N scan snapshots)
- `FKS_SNAPSHOT_RETENTION_DAYS` (prune scan snapshots older than N days)
   (You can also invoke pruning on-demand via `cargo run -- prune --keep <N> --max-age-days <D>`)
  
Snapshot metrics:

- `stats` subcommand surfaces aggregated compression metrics (supports `--format json`, `--limit N`).
- Report includes a "Compression Trend" table (last 10 snapshots) when metadata is present.
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (optional enrichment)

### 5. Running Bot

```bash
DISCORD_TOKEN=... cargo run --no-default-features --features discord -- bot

# Run HTTP server only (default feature already server)
cargo run -- serve --port 4802

# Run server + bot
DISCORD_TOKEN=... cargo run --features "server,discord" -- bot
```

Commands:

- `!ping` health check
- `!scan` quick stats
- `!suggest` prioritized task prompt (enrichment available via server endpoint `GET /suggest?enrich=1`)
- `!prompts` list prompt categories or `!prompts <Category>`
- `!prompt <name_part>` quick search prompt template by name substring (if exactly one match, returns filled)
- `!file <path>` retrieve file (truncated)

### 6. Hardening

- Add allowlists (guild/channel) before production usage.
- Rotate token if leaked (`Regenerate` in portal).
- Limit file return size via env.

## Environment File Example

## Structured Prompt Catalog & Placeholders

Unified prompt access paths:

- CLI list: `cargo run -- prompts [--category X] [--name substr] --format text|json|md`
- CLI single: `cargo run -- prompt --name "Exact Name" [--fill] --format text|json`
- HTTP list: `GET /prompts?category=X&name=substr&format=json|md|text`
- HTTP single: `GET /prompt?name=Exact%20Name&fill=1&format=json|text`
- Discord: `!prompts`, `!prompt <substring>`, `/prompts`, `/prompt name:<Exact Name> [fill:yes|no]`

Auto-filled placeholders when fill enabled:

| Placeholder    | Source                                                    |
|----------------|-----------------------------------------------------------|
| `[REPO_NAME]`  | Current repo directory name                               |
| `[LIST_REPOS]` | Sibling `fks_*` repo names (comma-separated)               |
| `[TOP_EXTS]`   | Top file extensions with counts (from quick scan)         |
| `[TOTAL_FILES]`| Approx total files counted via quick scan (sum of top ext counts) |
| `[PRIMARY_LANG]` | Primary language inferred from most common extension     |

Example (CLI):

```bash
cargo run -- prompt --name "Repo Analytics" --fill
```

Example (HTTP):

```bash
curl -s 'http://localhost:4802/prompt?name=Repo%20Analytics&fill=1' | sed -n '1,40p'
```

Example JSON (abridged):

```json
{
   "name": "Repo Analytics",
   "category": "Misc",
   "description": "Activity insights with extension context",
   "template_raw": "Suggest ways to analyze activity across [LIST_REPOS] ... [TOP_EXTS]. Total ~[TOTAL_FILES] files; primary language: [PRIMARY_LANG].",
   "filled": "Context: repo=fks_analyze files=NNN top_exts=[rs:120, toml:3]\nSuggest ways to analyze activity across fks_analyze, fks_api ... Observed top file extensions in current repo: rs:120, toml:3.",
   "guidelines": "..."
}
```

Unrecognized bracket tokens are left untouched so you can refine them interactively.

Catalog source: `src/prompts.rs` (consider syncing with `docs/prompts.md` for external editing in future automation).


See `.env.example` for a full template.

If you want to run a lightweight local enrichment backend, start the helper transformer service (Python) first or use the orchestration flag:

```bash
# From fks_analyze directory
./start.sh --with-transformer

# Or manually (in sibling fks_transformer dir):
docker compose up -d --build fks_transformer
```

Then ensure `OLLAMA_BASE_URL` in your `.env` points to `http://localhost:4500` (the default in `.env.example`). The `suggest --enrich` path and `!suggest+` (if added) will call the local endpoint.

## Roadmap (Condensed)

1. Advanced suggestions w/ diff scoring & hotspots
2. Slash command migration
3. Rich search (semantic / embeddings)
4. HTTP API & Websocket events
5. Test suite & CI workflows

## License

MIT
