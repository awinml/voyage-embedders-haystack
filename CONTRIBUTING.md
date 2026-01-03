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

    Note: This project requires Python 3.10 or higher.

3. The project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, ensure uv is installed:

   ```bash
   # Install uv (if not already installed)
   pip install uv
   ```

4. Sync the project and install all dependencies:

    ```bash
    make sync
    ```

    Or manually:

    ```bash
    uv sync --all-extras
    ```

    This creates an isolated virtual environment with project dependencies and the project installed in editable mode.

## Code Quality Standards

### Testing

Before submitting a pull request, ensure all tests pass:

```bash
make test
```

To run tests with coverage:

```bash
make cov
```

Add tests for any new functionality or bug fixes.

### Code Quality

This project uses:

- [Black](https://black.readthedocs.io/) for code formatting.
- [Ruff](https://docs.astral.sh/ruff/) for linting.
- [Mypy](https://mypy.readthedocs.io/) for type checking.

Format and lint your code before submitting:

```bash
make lint-all
```

This command will format code with Black, apply auto-fixes with Ruff, and check typing with MyPy.

### Useful Development Commands

- `make sync` - Create/sync development environment.
- `make test` - Run tests.
- `make cov` - Run tests with coverage.
- `make lint-all` - Run all code quality checks and formatting.
- `make lint-fmt` - Format code and apply auto-fixes.
- `make lint-style` - Lint only (no changes).
- `make lint-typing` - Type check only.

## Community Guidelines

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

By contributing to this project, you agree that your contributions will be licensed under the Apache-2.0 license as specified in the [LICENSE](LICENSE) file.
