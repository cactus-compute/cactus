---
title: "Contributing to Cactus"
description: "Guidelines for contributing to Cactus, the on-device AI inference engine. Covers code style, PR process, testing, and benchmarking."
keywords: ["contributing", "open source", "pull request", "code style", "C++20"]
---

# Contributing to Cactus

Thank you for your interest in contributing to Cactus! This document covers the guidelines and process for making contributions.

## Code Guidelines

- **C++ Standard**: Use C++20 features where appropriate.
- **Formatting**: Follow the existing code style in the project, one header per folder.
- **Comments**: Avoid comments, make your code read like plain english.
- **AI-Generated Code**: Do not blindly PR AI slop, this codebase is very complex, they miss details.
- **Update docs**: Please update docs when necessary, be intuitive and straightforward.
- **Keep It Simple**: Do not go beyond the scope of the GH issue, avoid bloated PRs, keep codes lean.
- **Benchmark Your Changes**: Test performance impact, Cactus is performance-critical.
- **Test everything**: A PR that fails to build is the biggest red flag, means it was not tested.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping the scope focused on the relevant GitHub issue.
3. Run `cactus test` to verify your changes build and pass all tests.
4. Run `cactus test --benchmark` if your changes affect performance-critical paths.
5. Update documentation if your changes affect the public API or user-facing behavior.
6. Submit a pull request with a clear description of what you changed and why.

## Testing

```bash
# Run all tests
cactus test

# Run tests on a connected iOS device
cactus test --ios

# Run tests on a connected Android device
cactus test --android

# Run benchmarks
cactus test --benchmark

# Test a specific model
cactus test --model LiquidAI/LFM2.5-1.2B-Instruct
```

## Developer Certificate of Origin

All contributions must comply with the [Developer Certificate of Origin (DCO)](DCO.md). By submitting a contribution, you certify that you have the right to do so under the project's open source license.

## See Also

- [Cactus Engine API](/docs/cactus_engine.md) — C API reference
- [Cactus Graph API](/docs/cactus_graph.md) — Computational graph API reference
- [Documentation Index](/docs/index.md) — Full documentation table of contents
