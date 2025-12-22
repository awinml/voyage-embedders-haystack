# Contributing to Voyage Haystack Integration

Thank you for your interest in contributing to Voyage Haystack Integration! We welcome contributions from the community and appreciate your efforts to improve this project.

## How to Contribute

There are many ways to contribute to the Voyage Haystack Integration:

- Submit bug reports or feature requests via [GitHub Issues](https://github.com/awinml/voyage-embedders-haystack/issues).
- Fix bugs or implement features via [Pull Requests](https://github.com/awinml/voyage-embedders-haystack/pulls).
- Improve documentation.
- Review existing pull requests.
- Help answer questions from other users.

## Development Setup

1. Fork the repository on GitHub.

2. Clone your fork locally and navigate to the project directory:

   ```bash
   cd voyage-embedders-haystack
   ```

3. Install [Hatch](https://hatch.pypa.io/) project manager:

   ```bash
   pip install hatch
   ```

4. Install the project in development mode:

   ```bash
   hatch shell
   ```

   This creates an isolated virtual environment with project dependencies and the project installed in editable mode.

## Code Quality Standards

### Testing

Before submitting a pull request, ensure all tests pass:

```bash
hatch run test
```

To run tests with coverage:

```bash
hatch run cov
```

Add tests for any new functionality or bug fixes.

### Code Quality

This project uses:

- [Black](https://black.readthedocs.io/) for code formatting.
- [Ruff](https://docs.astral.sh/ruff/) for linting.
- [Mypy](https://mypy.readthedocs.io/) for type checking.

Format and lint your code before submitting:

```bash
hatch run lint:all
```

This command will format code with Black, apply auto-fixes with Ruff, and check typing with MyPy.

### Useful Development Commands

- `hatch shell` - Activate development environment.
- `hatch run test` - Run tests.
- `hatch run cov` - Run tests with coverage.
- `hatch run lint:all` - Run all code quality checks.
- `hatch run lint:fmt` - Format code and apply auto-fixes.
- `hatch run lint:style` - Lint only.
- `hatch run lint:typing` - Type check only.

## Community Guidelines

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

By contributing to this project, you agree that your contributions will be licensed under the Apache-2.0 license as specified in the [LICENSE](LICENSE) file.
