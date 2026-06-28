# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial Rust port of [ResampleScope](http://entropymine.com/resamplescope/)
  with an in-memory callback API: `analyze` / `analyze_downscale` /
  `analyze_upscale`, filter scoring against the built-in `KnownFilter` set,
  edge-mode detection, scope-graph rendering, and a `reference` module
  (`perfect_resize`, `compute_weights`, `ssim`) (4794cd7).
- House-standard CI matrix — Linux/macOS/Windows on x64 and arm64, MSRV 1.89,
  i686 via `cross`, the `c-reference` feature, lint, docs, and coverage — plus
  README badges (2fb4673).

### Changed

- README overhaul to the shared zen conventions: a `Quick start` built around
  `analyze`, an MSRV badge, documented `ssim` and single-direction analysis, and
  the shared crosslink footer. Split the crates.io README into a generated
  `README.crates.md` (CI badge only) and pointed `readme` at it.
