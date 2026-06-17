use assert_cmd::Command;

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
