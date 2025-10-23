# Contributing to PyFastL2LiR

Thank you for your interest in contributing to PyFastL2LiR! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv)

## Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/<USERNAME>/PyFastL2LiR.git
   cd PyFastL2LiR
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   uv sync
   ```

   This will install the package in editable mode along with all development dependencies.

## Making Changes

1. **Create a new branch for your changes**

   ```bash
   git checkout -b feature/your-feature-name
   ```

   Use descriptive branch names:
   - `feature/` for new features
   - `fix/` for bug fixes
   - `docs/` for documentation changes
   - `refactor/` for code refactoring

2. **Make your changes**

   - Write clear, concise commit messages
   - Keep commits focused and atomic
   - Update documentation as needed

## Testing

We use pytest for testing. All tests should pass before submitting a pull request.

### Running Tests

Run all tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=fastl2lir --cov-report=term-missing
```

Run specific tests:

```bash
pytest tests/test_specific_module.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with the `test_` prefix
- Name test functions with the `test_` prefix
- Aim for high test coverage, especially for new features
- Include both unit tests and integration tests where appropriate

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting.

### Running Ruff

Check for linting issues:

```bash
ruff check .
```

Automatically fix issues:

```bash
ruff check --fix .
```

Format code:

```bash
ruff format .
```

### Before Committing

Make sure your code passes both linting and formatting checks:

```bash
ruff check . && ruff format --check .
```

## Submitting Changes

1. **Ensure all tests pass**

   ```bash
   pytest
   ```

2. **Ensure code passes linting**

   ```bash
   ruff check .
   ruff format --check .
   ```

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "[tag] Brief description of your changes"
   ```

   Write clear commit messages following these guidelines:
   - Use the imperative mood ("Add feature" not "Added feature")
   - Provide additional details in the body if needed

4. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes
   - Reference any related issues

### Pull Request Guidelines

- Provide a clear description of the problem and solution
- Include relevant issue numbers (e.g., "Fixes #123")
- Ensure CI checks pass
- Respond to review comments promptly
- Keep pull requests focused on a single feature or fix

## Code Review Process

- Maintainers will review your pull request
- Address any requested changes
- Once approved, your changes will be merged

## Questions?

If you have questions or need help, feel free to:
- Open an issue on GitHub
- Reach out to the maintainers

## License

By contributing to PyFastL2LiR, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing!
