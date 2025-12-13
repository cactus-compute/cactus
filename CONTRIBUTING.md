# Contributing to Cactus

Thank you for your interest in contributing to Cactus! This document provides guidelines and instructions for contributing to this project.

## Developer Certificate of Origin (DCO)

This project uses the Developer Certificate of Origin (DCO) to ensure that all contributions are properly licensed and that contributors have the right to submit their work.

### What is the DCO?

The DCO is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing to the project. It is used by many open source projects including the Linux kernel.

### DCO Sign-Off

All commits must be signed off to indicate that you agree to the terms of the [DCO](https://developercertificate.org/):

### How to Sign-Off Your Commits

You have two options for signing off your commits:

#### Option 1: Automatic Sign-off (Recommended)

Set up automatic DCO sign-off by running our setup script once after cloning:

```bash
# Run once after cloning the repository
./tools/setup-dco.sh

# Then commit normally - sign-off is added automatically
git commit -m "Your commit message"
```

#### Option 2: Manual Sign-off

Manually sign-off each commit using the `-s` or `--signoff` flag:

```bash
git commit -s -m "Your commit message"
```

This will add a `Signed-off-by` line at the end of your commit message:

```
Your commit message

Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your Git configuration. You can set these with:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Fixing Unsigned Commits

If you've already made commits without signing off, you can amend the last commit:

```bash
git commit --amend -s
```

For multiple commits, you can rebase and sign them:

```bash
git rebase HEAD~3 --signoff  # Sign-off last 3 commits
```

## Contribution Guidelines

### Code Style

- **C++ Standard**: Use C++20 features where appropriate
- **Formatting**: Follow the existing code style in the project
- **ARM NEON**: When writing SIMD code, ensure proper alignment and use appropriate intrinsics
- **Comments**: Add comments for complex algorithms, especially in kernel implementations

### Performance Considerations

Cactus is a high-performance inference library optimized for ARM processors. When contributing:

1. **Benchmark Your Changes**: Test performance impact, especially for kernel functions
2. **Memory Efficiency**: Minimize memory allocations in hot paths
3. **SIMD Optimization**: Use ARM NEON intrinsics where beneficial
4. **Cache Awareness**: Consider cache line sizes and memory access patterns

### Testing

Before submitting a PR:

1. Ensure all existing tests pass
2. Add tests for new functionality
3. Test on ARM hardware if possible (Apple Silicon, Raspberry Pi, etc.)
4. Verify quantized operations maintain acceptable accuracy

### Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Make your changes** following the guidelines above
3. **Sign-off all commits** using DCO
4. **Update documentation** if you change APIs
5. **Open a Pull Request** with a clear title and description

#### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Tested on ARM hardware
- [ ] Benchmarked performance impact

## Checklist
- [ ] All commits are signed-off (DCO)
- [ ] Code follows project style
- [ ] Comments added where necessary
- [ ] Documentation updated if needed
```

### Reporting Issues

When reporting issues, please include:

1. System information (OS, CPU architecture, ARM variant)
2. Cactus version or commit hash
3. Minimal code to reproduce the issue
4. Expected vs actual behavior
5. Any relevant logs or error messages

### Areas of Contribution

We especially welcome contributions in these areas:

- **Kernel Optimizations**: SIMD implementations for ARM architectures
- **Quantization**: Improved quantization techniques (INT8, INT4)
- **Model Support**: Support for additional model architectures
- **NPU Integration**: Apple Neural Engine and other NPU backends
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: Test coverage and benchmarking infrastructure

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cactus.git
cd cactus
```

You can run these codes directly on M-series Macbooks since they are ARM-based.
Vanilla M3 CPU-only can run Qwen3-600m-INT8 at 60-70 toks/sec, just run the following: 

```bash
./tests/run.sh # chmod +x first time
```

## Generating weights from HuggingFace 
Use any of the following (270m, 600m, 1B, 1.7B activated params):
```bash
python3 tools/convert_hf.py google/gemma-3-270m-it weights/gemma3-270m-i8/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m-i8/ --precision INT8
python3 tools/convert_hf.py google/gemma-3-1b-it weights/gemma3-1b-i8/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-1.7B weights/qwen3-1.7-i8/ --precision INT8
```

Simply replace the weight path `tests/test_engine.cpp` with your choice.

### Questions?

If you have questions about contributing, feel free to:

1. Open an issue for discussion
2. Check existing issues and PRs
3. Review the codebase documentation

## License

By contributing to Cactus, you agree that your contributions will be licensed under the same license as the project (check LICENSE file).

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

Thank you for contributing to Cactus! 