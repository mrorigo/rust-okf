use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn cli_can_write_default_config() {
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("okf.toml");

    Command::cargo_bin("rust-okf")
        .unwrap()
        .args(["--config", config.to_str().unwrap(), "init-config"])
        .assert()
        .success()
        .stdout(contains("wrote default config"));

    assert!(config.exists());
}
