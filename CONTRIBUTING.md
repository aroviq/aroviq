# Contributing to Aroviq

Thank you for your interest in improving Aroviq! We welcome contributions ranging from bug fixes, documentation improvements, to new security verifiers.

## Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/arovq/aroviq.git
   cd aroviq
   ```
2. **Install Poetry:**
   If you don't have Poetry installed, see the [Poetry Docs](https://python-poetry.org/docs/#installation).
3. **Install Dependencies:**
   ```bash
   poetry install
   ```
   This will install all package dependencies and development tools (`pytest`, `ruff`, `mypy`).

## Running Tests

Before submitting a Pull Request, please ensure all tests pass:

```bash
poetry run pytest tests/
```

We require any code changing core engine components or verifiers to write associated unit tests.

## Code Quality

We use `ruff` to enforce our style guide.

```bash
poetry run ruff check .
poetry run ruff format .
```

And `mypy` for static type checking:

```bash
poetry run mypy .
```

## Submitting a Pull Request

1.  Fork the repo and create your feature branch from `main`.
2.  If you've added new features, ensure tests describe the exact behavior.
3.  Update the documentation as needed.
4.  Submit a Pull Request with a clear description of your changes! We aim to review PRs within 48 hours.

Thank you for helping us make autonomous AI agents safer.
