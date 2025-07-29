from fnmatch import fnmatch
from pathlib import Path

import aiofiles


def should_ignore_path(path: Path, name: str) -> bool:
    """Check if a path should be ignored based on common patterns."""
    ignore_patterns = [
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        ".DS_Store",
        ".env",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dll",
        "*.exe",
        "*.bin",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.pdf",
        "*.zip",
        "*.tar",
        "*.gz",
    ]

    path_str = str(path)

    for pattern in ignore_patterns:
        if pattern.startswith("*."):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif pattern in path_str:
            return True

    return False


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


async def read_file_content(path: Path) -> tuple[str | None, Exception | None]:
    """Read file content and handle errors."""
    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            content = await f.read()
        return content, None
    except Exception as e:
        return None, e
