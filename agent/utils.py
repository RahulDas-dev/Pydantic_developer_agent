from pathlib import Path


def is_git_repository(path: str | Path) -> bool:
    """
    Check if a directory is a Git repository.

    Args:
        path: Directory path to check

    Returns:
        True if the directory is a Git repository
    """
    git_dir = Path(path) / ".git"
    return git_dir.exists() and (git_dir.is_dir() or git_dir.is_file())


PYTHON_FILES = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "environment.yml", "conda.yml"]


def has_python_files(workspace_path: Path | str) -> bool:
    """
    Check if the given workspace path contains any Python files.
    """
    return any(Path(workspace_path, file).exists() for file in PYTHON_FILES)


def is_valid_workspace(workspace_path: Path | str) -> bool:
    """
    Check if the given workspace path is valid.
    A valid workspace must contain at least one Python file.
    """
    workspace_path = Path(workspace_path)
    return workspace_path.is_dir()
