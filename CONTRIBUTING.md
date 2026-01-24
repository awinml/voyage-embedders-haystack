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

- [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting.
- [ty](https://docs.astral.sh/ty/) for type checking (Astral's Rust-based type checker).

Format and lint your code before submitting:

```bash
make lint-all
```

This command will format code with Ruff, apply auto-fixes with Ruff, and check typing with ty.

### Useful Development Commands

- `make sync` - Create/sync development environment.
- `make test` - Run tests.
- `make cov` - Run tests with coverage.
- `make lint-all` - Run all code quality checks and formatting.
- `make lint-fmt` - Format code and apply auto-fixes.
- `make lint-style` - Lint only (no changes).
- `make lint-typing` - Type check only.

## Releasing a New Version

This project follows [Semantic Versioning](https://semver.org/). Releases are automated via GitHub Actions when a version tag is pushed.

### Preparation Steps

1. **Create a release branch** from `main`:

   ```bash
   git checkout -b release/v1.x.y
   ```

2. **Update the version** in `pyproject.toml`:

   ```toml
   version = "1.x.y"
   ```

3. **Update the README** with a new entry in the "What's New" section:

   ```markdown
   - **[v1.x.y - MM/DD/YY]:**
     - Brief description of major changes
   ```

4. **Run all checks** to ensure quality:

   ```bash
   make lint-all  # Format, lint, and type check
   make test      # Run all tests
   ```

5. **Commit and push** these changes:

   ```bash
   git commit -am "chore: Bump version to 1.x.y"
   git push origin release/v1.x.y
   ```

### Creating the Release

#### **Option 1: GitHub Web UI (Recommended for simplicity)**

1. **Create a Pull Request** on GitHub and ensure all CI checks pass.

2. **Merge the PR** to `main` once approved.

3. **Create Release via GitHub**:
   - Go to [Releases](https://github.com/awinml/voyage-embedders-haystack/releases) page
   - Click "Create a new release"
   - Click "Choose a tag" and type `v1.x.y` (GitHub will create the tag automatically)
   - Fill in the release title: `Version 1.x.y`
   - Add release notes/description describing major changes
   - Click "Publish release"

#### **Option 2: Command Line (For automation/scripting)**

1. **Create a Pull Request** on GitHub and ensure all CI checks pass.

2. **Merge the PR** to `main` once approved.

3. **Create a git tag** on the `main` branch:

   ```bash
   git checkout main
   git pull origin main
   git tag v1.x.y
   git push origin --tags
   ```

4. **Create Release on GitHub** (optional, for documentation):
   - The tag push triggers the workflow automatically
   - You can manually create a release afterwards for release notes, or use GitHub CLI:

   ```bash
   gh release create v1.x.y --title "Version 1.x.y" --notes "Release notes here"
   ```

### Automated Deployment

When a tag matching the pattern `v[0-9].[0-9]+.[0-9]+*` is created (via either method):

- GitHub Actions will automatically:
  1. Build the source distribution and wheel using uv
  2. Publish to PyPI using Trusted Publishing
  3. Make the release publicly available (check in 2-5 minutes)

### Versioning Guidelines

- **Patch release** (1.x.Y): Bug fixes, documentation updates
- **Minor release** (1.X.0): New features, backward compatible
- **Major release** (X.0.0): Breaking changes, API modifications

### Verifying the Release

After the GitHub Actions workflow completes:

1. Check [PyPI](https://pypi.org/project/voyage-embedders-haystack/) to confirm the release is published.
2. Verify the package can be installed: `pip install voyage-embedders-haystack==1.x.y`.
3. Check the [GitHub Releases](https://github.com/awinml/voyage-embedders-haystack/releases) page.

## Community Guidelines

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

By contributing to this project, you agree that your contributions will be licensed under the Apache-2.0 license as specified in the [LICENSE](LICENSE) file.
