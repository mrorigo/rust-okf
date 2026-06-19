# rust-okf Release Process

This document describes how to produce a GitHub release for `rust-okf`.

## Release model

Releases are tag-driven.

Pushing a tag that matches `v*.*.*` triggers the GitHub Actions release workflow in `.github/workflows/release.yml`.

The workflow:

- builds the `okf` binary for the configured target matrix
- packages release artifacts
- uploads artifacts to GitHub Actions
- publishes a GitHub Release with generated notes

## Before you release

Make sure the repository is in a releasable state.

Run:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

If you changed performance-sensitive code, also run:

```bash
cargo bench --bench operations
```

Review the results and update the benchmark table in `README.md` if the baseline changed materially.

## Versioning

`rust-okf` uses the crate version in `Cargo.toml` as the release version.

Before tagging:

- bump `[package].version` in `Cargo.toml`
- ensure the changelog or release notes reflect the new version
- verify the CLI binary name remains `okf`

Use a semantic version tag in the form:

```text
vX.Y.Z
```

Examples:

- `v0.1.1`
- `v0.2.0`
- `v1.0.0`

## Release checklist

1. Update the version in `Cargo.toml`.
2. Run the test, lint, and formatting checks.
3. Run benchmarks if relevant.
4. Commit the release changes.
5. Create a signed tag if your workflow uses signed tags.
6. Push the tag to GitHub.
7. Verify that the GitHub Actions release workflow starts.
8. Confirm the GitHub Release is created and artifacts are attached.

## Tag push

The simplest release command sequence is:

```bash
git tag v0.1.1
git push origin v0.1.1
```

If you need to create the tag after the version bump commit is already pushed, make sure the tag points to the intended commit.

## What GitHub publishes

The release workflow creates artifacts for the supported targets in the release matrix.

The published files are packaged from the `okf` binary, not the library crate.

Expected artifact names follow the target triple, for example:

- `okf-x86_64-unknown-linux-gnu.tar.gz`
- `okf-aarch64-apple-darwin.tar.gz`
- `okf-x86_64-pc-windows-msvc.zip`

Each artifact also gets a SHA-256 checksum file.

## Verification after release

After the release completes:

- open the GitHub Release page
- confirm the version tag is correct
- confirm release notes were generated
- confirm all target artifacts are present
- download at least one artifact and verify the binary runs

If you changed the command-line interface, verify the release artifact matches the expected binary name:

```bash
okf --help
```

## Failure handling

If the release workflow fails:

- inspect the GitHub Actions logs
- confirm the tag matches `v*.*.*`
- confirm the workflow file paths are correct
- confirm the version in `Cargo.toml` matches the intended release

If a bad tag was pushed, do not repurpose it. Create a new tag with the corrected version.

## Notes

- The release workflow does not publish to crates.io.
- The release workflow is separate from CI.
- Tag pushes are the source of truth for release publication.
