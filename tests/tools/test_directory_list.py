# ruff: noqa: S101
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from lib.context import AgentContext
from lib.tools.directory_list import (
    DirectoryEntry,
    DirectoryInfo,
    _list_directory_recursive,
    list_directory,
)

# Constants for test expectations
MIN_CONTENT_LINES = 3  # Header + stats line + empty line
TEST_ENTRY_NAME = "test.txt"
TEST_ENTRY_PATH = "/path/to/test.txt"
TEST_ENTRY_DEPTH = 2
TEST_ENTRY_SIZE = 1024

TEST_INFO_PATH = "/test/path"
TEST_INFO_ENTRIES = 10
TEST_INFO_DIRS = 3
TEST_INFO_FILES = 7

# Test directory structure constants
EXPECTED_TOP_LEVEL_ENTRIES = 3  # file1.txt, file2.txt, and subdir1
EXPECTED_TOP_LEVEL_WITH_HIDDEN = 5  # includes .hidden_file and .hidden_dir
EXPECTED_DIR_INFO_ENTRIES = 3  # file1.txt, file2.txt, and subdir1
EXPECTED_DIR_INFO_DIRS = 1  # subdir1
EXPECTED_DIR_INFO_FILES = 2  # file1.txt, file2.txt


def test_directory_entry_model() -> None:
    """Test the DirectoryEntry model."""
    entry = DirectoryEntry(
        name=TEST_ENTRY_NAME,
        path=TEST_ENTRY_PATH,
        is_dir=False,
        depth=TEST_ENTRY_DEPTH,
        size=TEST_ENTRY_SIZE,
    )

    assert entry.name == TEST_ENTRY_NAME
    assert entry.path == TEST_ENTRY_PATH
    assert not entry.is_dir
    assert entry.depth == TEST_ENTRY_DEPTH
    assert entry.size == TEST_ENTRY_SIZE


def test_directory_info_model() -> None:
    """Test the DirectoryInfo model."""
    info = DirectoryInfo(
        path=TEST_INFO_PATH,
        total_entries=TEST_INFO_ENTRIES,
        directories=TEST_INFO_DIRS,
        files=TEST_INFO_FILES,
    )

    assert info.path == TEST_INFO_PATH
    assert info.total_entries == TEST_INFO_ENTRIES
    assert info.directories == TEST_INFO_DIRS
    assert info.files == TEST_INFO_FILES


@pytest.fixture
def temp_dir_structure() -> Generator[Path, None, None]:
    """Create a temporary directory structure for testing."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create some regular files
        (root / "file1.txt").write_text("test content")
        (root / "file2.txt").write_text("more content")

        # Create a hidden file
        (root / ".hidden_file").write_text("hidden content")

        # Create subdirectories
        sub_dir1 = root / "subdir1"
        sub_dir1.mkdir()
        (sub_dir1 / "subfile1.txt").write_text("subfile content")

        # Create nested subdirectory
        sub_dir2 = sub_dir1 / "nested"
        sub_dir2.mkdir()
        (sub_dir2 / "deep_file.txt").write_text("deep content")

        # Create hidden directory
        hidden_dir = root / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "hidden_subfile.txt").write_text("hidden subfile")

        yield root


async def _run_list_recursive_test(
    root: Path, show_hidden: bool, recursive: bool, max_depth: int
) -> List[DirectoryEntry]:
    """Helper function to run the list_directory_recursive function."""
    return await _list_directory_recursive(root, show_hidden, recursive, max_depth, 0)


@pytest.mark.asyncio
async def test_basic_listing_no_hidden(temp_dir_structure: Path) -> None:
    """Test basic directory listing without hidden files."""
    entries = await _run_list_recursive_test(temp_dir_structure, False, False, 1)

    # Should only include file1.txt, file2.txt, and subdir1/
    assert len(entries) == EXPECTED_TOP_LEVEL_ENTRIES

    # Check if the right files and directories are included
    names = {entry.name for entry in entries}
    assert "file1.txt" in names
    assert "file2.txt" in names
    assert "subdir1" in names
    assert ".hidden_file" not in names
    assert ".hidden_dir" not in names

    # Verify all entries are at depth 0
    for entry in entries:
        assert entry.depth == 0


@pytest.mark.asyncio
async def test_with_hidden_files(temp_dir_structure: Path) -> None:
    """Test listing with hidden files included."""
    entries = await _run_list_recursive_test(temp_dir_structure, True, False, 1)

    # Should include all top-level items
    assert len(entries) == EXPECTED_TOP_LEVEL_WITH_HIDDEN

    # Check if hidden files and directories are included
    names = {entry.name for entry in entries}
    assert ".hidden_file" in names
    assert ".hidden_dir" in names


@pytest.mark.asyncio
async def test_recursive_listing(temp_dir_structure: Path) -> None:
    """Test recursive directory listing."""
    entries = await _run_list_recursive_test(temp_dir_structure, False, True, 3)

    # Collect all paths to verify the structure
    paths = {entry.path for entry in entries}

    # Verify files at various depths
    assert str(temp_dir_structure / "file1.txt") in paths
    assert str(temp_dir_structure / "subdir1") in paths
    assert str(temp_dir_structure / "subdir1" / "subfile1.txt") in paths
    assert str(temp_dir_structure / "subdir1" / "nested") in paths
    assert str(temp_dir_structure / "subdir1" / "nested" / "deep_file.txt") in paths


@pytest.mark.asyncio
async def test_depth_limitation(temp_dir_structure: Path) -> None:
    """Test that max_depth limits the recursion properly."""
    # Set max_depth to 1 - should not see files in nested directory
    entries = await _run_list_recursive_test(temp_dir_structure, False, True, 1)

    paths = {entry.path for entry in entries}

    # These should be included
    assert str(temp_dir_structure / "file1.txt") in paths
    assert str(temp_dir_structure / "subdir1" / "subfile1.txt") in paths

    # This should NOT be included due to depth limitation
    assert str(temp_dir_structure / "subdir1" / "nested" / "deep_file.txt") not in paths


@pytest.mark.asyncio
async def test_permission_error_handling() -> None:
    """Test handling of permission errors."""
    test_path = Path("/test/path")

    # Mock Path.iterdir to raise PermissionError
    with (
        patch("pathlib.Path.iterdir", side_effect=PermissionError("Access denied")),
        patch("lib.tools.directory_list.logger") as mock_logger,
    ):
        entries = await _list_directory_recursive(test_path, False, False, 3, 0)

        # Should return empty list on permission error
        assert len(entries) == 0

        # Should log the error
        mock_logger.debug.assert_called_with(f"Permission denied listing directory: {test_path}")


@pytest.fixture
def mock_agent_context(temp_dir_structure: Path) -> MagicMock:
    """Create a mock agent context for testing."""
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = MagicMock(spec=AgentContext)
    mock_ctx.deps.workspace_path = str(temp_dir_structure)
    return mock_ctx


@pytest.mark.asyncio
async def test_basic_directory_listing(temp_dir_structure: Path, mock_agent_context: MagicMock) -> None:
    """Test basic directory listing functionality."""
    result = await list_directory(mock_agent_context, str(temp_dir_structure))

    # Verify it's a ToolReturn object
    assert isinstance(result, ToolReturn)

    # Verify success metadata
    assert result.metadata.get("success") is True

    # Verify directory info in metadata
    dir_info = result.metadata.get("directory_info")
    assert dir_info is not None
    assert dir_info["total_entries"] == EXPECTED_DIR_INFO_ENTRIES
    assert dir_info["directories"] == EXPECTED_DIR_INFO_DIRS
    assert dir_info["files"] == EXPECTED_DIR_INFO_FILES

    # Verify content structure
    assert isinstance(result.content, list)
    assert len(result.content) > MIN_CONTENT_LINES

    # Check header in content
    assert any(line.startswith("# Directory listing for:") for line in result.content)


@pytest.mark.asyncio
async def test_directory_not_found(mock_agent_context: MagicMock) -> None:
    """Test error handling when directory doesn't exist."""
    with patch("pathlib.Path.exists", return_value=False):
        result = await list_directory(mock_agent_context, "/non/existent/path")

        # Verify error metadata
        assert result.metadata.get("success") is False

        # Verify error message
        assert "Directory not found" in result.return_value
        assert any("Error: Directory Not Found" in line for line in result.content)


@pytest.mark.asyncio
async def test_not_a_directory(temp_dir_structure: Path, mock_agent_context: MagicMock) -> None:
    """Test error handling when path is not a directory."""
    file_path = temp_dir_structure / "file1.txt"

    with patch("pathlib.Path.is_dir", return_value=False):
        result = await list_directory(mock_agent_context, str(file_path))

        # Verify error metadata
        assert result.metadata.get("success") is False

        # Verify error message
        assert "Path is not a directory" in result.return_value
        assert any("Error: Not a Directory" in line for line in result.content)


@pytest.mark.asyncio
async def test_permission_error(temp_dir_structure: Path, mock_agent_context: MagicMock) -> None:
    """Test permission error handling."""
    with patch("lib.tools.directory_list._list_directory_recursive", side_effect=PermissionError("Access denied")):
        result = await list_directory(mock_agent_context, str(temp_dir_structure))

        # Verify error metadata
        assert result.metadata.get("success") is False
        assert result.metadata.get("error") == "PermissionError"

        # Verify error message
        assert "Permission denied" in result.return_value
        assert any("Error: Permission Denied" in line for line in result.content)


@pytest.mark.asyncio
async def test_general_exception(temp_dir_structure: Path, mock_agent_context: MagicMock) -> None:
    """Test general exception handling."""
    with patch("lib.tools.directory_list._list_directory_recursive", side_effect=ValueError("Something went wrong")):
        result = await list_directory(mock_agent_context, str(temp_dir_structure))

        # Verify error metadata
        assert result.metadata.get("success") is False
        assert result.metadata.get("error") == "ValueError"

        # Verify error message
        assert "Failed to list directory" in result.return_value
        assert any("Error Listing Directory" in line for line in result.content)


@pytest.mark.asyncio
async def test_max_depth_validation(temp_dir_structure: Path, mock_agent_context: MagicMock) -> None:
    """Test validation of max_depth parameter."""
    # Test with value below minimum
    with (
        patch("lib.tools.directory_list.logger"),
        patch("lib.tools.directory_list._list_directory_recursive", return_value=[]),
    ):
        result = await list_directory(mock_agent_context, str(temp_dir_structure), max_depth=0)
        # Should be adjusted to minimum value (1)
        assert result.metadata.get("success") is True

    # Test with value above maximum
    with (
        patch("lib.tools.directory_list.logger"),
        patch("lib.tools.directory_list._list_directory_recursive", return_value=[]),
    ):
        result = await list_directory(mock_agent_context, str(temp_dir_structure), max_depth=20)
        # Should be adjusted to maximum value (10)
        assert result.metadata.get("success") is True
