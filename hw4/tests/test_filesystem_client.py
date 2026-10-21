"""Tests for the published Filesystem MCP server client."""

from __future__ import annotations

import unittest
from pathlib import Path

from study_assistant.filesystem_client import (
    filesystem_server_parameters,
    verify_filesystem_server,
)


class FilesystemConfigurationTests(unittest.TestCase):
    """Ensure the server receives one absolute allowlisted directory."""

    def test_server_is_limited_to_absolute_study_files_path(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        parameters = filesystem_server_parameters(project_root)

        allowed_directory = Path(parameters.args[-1])
        self.assertTrue(allowed_directory.is_absolute())
        self.assertEqual(allowed_directory, (project_root / "study_files").resolve())
        self.assertEqual(parameters.cwd, project_root.resolve())
        self.assertEqual(parameters.args[:-1], [
            "--no-install",
            "mcp-server-filesystem",
        ])

    def test_missing_study_files_directory_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            filesystem_server_parameters(Path(__file__).parent / "missing-project")


class FilesystemIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the installed npm server through a real MCP stdio connection."""

    async def test_discovers_tools_and_reads_sample_with_scoped_access(self) -> None:
        result = await verify_filesystem_server()

        self.assertIn("read_text_file", result.tools)
        self.assertIn("list_allowed_directories", result.tools)
        self.assertEqual(
            result.allowed_directories,
            (str((Path(__file__).resolve().parents[1] / "study_files").resolve()),),
        )
        self.assertIn("Cell Biology: Quick Review", result.sample_content)
        self.assertIn("Mitochondria", result.sample_content)
