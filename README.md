# Python Development Environment with uv and Ruff

This repository provides a modern Python development environment using VSCode's Dev Container feature, with uv for package management and Ruff for code formatting/linting.

## Features

### Core Components
- 🐳 Containerized development environment using VSCode Dev Containers
- 🚀 Fast package management with [uv](https://github.com/astral-sh/uv)
- ✨ Code formatting and linting with [Ruff](https://github.com/astral-sh/ruff)
- 🔍 Pre-commit hooks for code quality
- 🔄 GitHub Actions for CI/CD
- 🐍 Python 3.12 by default

### Development Environment

The development container is configured to provide a consistent and efficient Python development experience:

- **Fast Package Management**: Uses `uv` instead of pip for significantly faster dependency management
- **Modern Code Quality Tools**: Integrated Ruff for both formatting and linting
- **Git Integration**: Automatic mounting of local Git and SSH configurations
- **Pre-configured VSCode**: Essential extensions and settings for Python development
- **CI/CD Ready**: GitHub Actions workflows for testing, linting, and Docker builds

## Getting Started

### Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker](https://www.docker.com/products/docker-desktop)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [uv](https://docs.astral.sh/uv/installation/)

### Quick Start

1. Click "Use this template" on GitHub to create your repository
2. Clone your new repository
3. Open in VSCode
4. When prompted, click "Reopen in Container"
5. Run the setup script:
   ```bash
   python setup.py
   ```
6. Enter your project name when prompted

### Package Management

Use uv to manage Python packages. Since the project is already initialized, then from the root of the project run:

```bash
uv sync
```

To run python code, use the following command:

```bash
uv run python <path_to_script>
```

To run tests, use the following command:

```bash
uv test
```

### Formatting and Linting

To format code, use the following command:

```bash
uv run ruff check --fix .
```
