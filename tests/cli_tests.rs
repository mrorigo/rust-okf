use assert_cmd::Command;
use serde_json::Value;
use std::fs;

fn setup_index() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = tmp.path().join("bundle");
    let index = tmp.path().join("index");
    fs::create_dir_all(&bundle).unwrap();
    fs::write(
        bundle.join("mcp.md"),
        "---\ntype: Concept\ntitle: MCP Configuration\ntags:\n  - protocol\n---\nConfigure MCP servers and tools for agents.\n",
    )
    .unwrap();
    fs::write(
        bundle.join("other.md"),
        "---\ntype: Concept\ntitle: Unrelated Document\n---\nA document about something else.\n",
    )
    .unwrap();

    Command::cargo_bin("okf")
        .unwrap()
        .args(["--mock-embeddings", "--index"])
        .arg(&index)
        .args(["add"])
        .arg(&bundle)
        .assert()
        .success();

    (tmp, index)
}

#[test]
fn cli_can_write_default_config() {
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("okf.toml");

    Command::cargo_bin("okf")
        .unwrap()
        .args(["--config", config.to_str().unwrap(), "init-config"])
        .assert()
        .success();

    assert!(config.exists());
}

#[test]
fn search_defaults_to_concise_human_output() {
    let (_tmp, index) = setup_index();
    let assert = Command::cargo_bin("okf")
        .unwrap()
        .args(["--mock-embeddings", "--index"])
        .arg(&index)
        .args(["search", "mcp"])
        .assert()
        .success();
    let output = assert.get_output();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(stdout.contains("Search: mcp"));
    assert!(stdout.contains("MCP Configuration"));
    assert!(!stdout.contains("lexical_candidates"));
    assert!(!stderr.contains("lexical_candidates"));
}

#[test]
fn search_json_outputs_result_envelope_without_plan() {
    let (_tmp, index) = setup_index();
    let assert = Command::cargo_bin("okf")
        .unwrap()
        .args(["--mock-embeddings", "--index"])
        .arg(&index)
        .args(["search", "mcp", "--json"])
        .assert()
        .success();
    let output = assert.get_output();
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();

    assert!(value.get("results").is_some());
    assert!(value.get("total_hits").is_some());
    assert!(value.get("plan").is_none());
    assert!(output.stderr.is_empty());
}

#[test]
fn search_explain_keeps_plan_on_stderr() {
    let (_tmp, index) = setup_index();
    let assert = Command::cargo_bin("okf")
        .unwrap()
        .args(["--mock-embeddings", "--index"])
        .arg(&index)
        .args(["search", "mcp", "--explain"])
        .assert()
        .success();
    let output = assert.get_output();
    let plan: Value = serde_json::from_slice(&output.stderr).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Search: mcp"));
    assert!(plan.get("lexical_candidates").is_some());
    assert!(plan.get("vector_candidates").is_some());
    assert!(plan.get("fused").is_some());
}

#[test]
fn search_json_and_explain_split_streams() {
    let (_tmp, index) = setup_index();
    let assert = Command::cargo_bin("okf")
        .unwrap()
        .args(["--mock-embeddings", "--index"])
        .arg(&index)
        .args(["search", "mcp", "--json", "--explain"])
        .assert()
        .success();
    let output = assert.get_output();

    let results: Value = serde_json::from_slice(&output.stdout).unwrap();
    let plan: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert!(results.get("results").is_some());
    assert!(plan.get("query").is_some());
}
