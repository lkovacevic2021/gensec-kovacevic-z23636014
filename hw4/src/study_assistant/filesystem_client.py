"""Client helpers for the published MCP Filesystem server."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp import Client, MCPError
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_FILES_DIRECTORY = PROJECT_ROOT / "study_files"
SAMPLE_DOCUMENT = STUDY_FILES_DIRECTORY / "cell-biology.md"
FILESYSTEM_PACKAGE_VERSION = "2.0.0"


class FilesystemServerError(RuntimeError):
    """Raised when the published Filesystem MCP server cannot be used."""


@dataclass(frozen=True)
class FilesystemVerification:
    """Results from discovering and reading through the Filesystem MCP server."""

    tools: tuple[str, ...]
    allowed_directories: tuple[str, ...]
    sample_content: str


def _parse_allowed_directories(report: str) -> tuple[str, ...]:
    """Normalize the directory paths returned by the server's reporting tool."""
    directories = {
        str(Path(line.strip().removeprefix("-").strip()).resolve())
        for line in report.splitlines()
        if line.strip() and line.strip().casefold() != "allowed directories:"
    }
    return tuple(sorted(directories))


def filesystem_server_parameters(
    project_root: str | Path = PROJECT_ROOT,
) -> StdioServerParameters:
    """Build launch settings limited to this project's ``study_files`` folder."""
    root = Path(project_root).resolve()
    allowed_directory = (root / "study_files").resolve()
    if not allowed_directory.is_dir():
        raise FileNotFoundError(
            f"Study files directory does not exist: {allowed_directory}"
        )

    return StdioServerParameters(
        command="npx",
        args=[
            "--no-install",
            "mcp-server-filesystem",
            str(allowed_directory),
        ],
        cwd=root,
    )


def _tool_text(result: CallToolResult, tool_name: str) -> str:
    """Extract text returned by a server tool, raising on tool-level failures."""
    if result.is_error:
        details = "\n".join(
            item.text for item in result.content if isinstance(item, TextContent)
        )
        raise FilesystemServerError(
            f"Filesystem tool {tool_name!r} failed: {details or 'no error details'}"
        )

    text = "\n".join(
        item.text for item in result.content if isinstance(item, TextContent)
    )
    if not text:
        raise FilesystemServerError(
            f"Filesystem tool {tool_name!r} returned no text content."
        )
    return text


async def verify_filesystem_server() -> FilesystemVerification:
    """Launch the npm server, list its tools, and read the sample study document."""
    parameters = filesystem_server_parameters()
    sample_path = SAMPLE_DOCUMENT.resolve()

    try:
        async with Client(parameters, raise_exceptions=True) as client:
            tools = tuple(tool.name for tool in (await client.list_tools()).tools)
            required_tools = {"read_text_file", "list_allowed_directories"}
            missing_tools = required_tools.difference(tools)
            if missing_tools:
                raise FilesystemServerError(
                    "Filesystem server is missing expected tools: "
                    + ", ".join(sorted(missing_tools))
                )

            allowed_report = _tool_text(
                await client.call_tool("list_allowed_directories"),
                "list_allowed_directories",
            )
            expected_directory = str(STUDY_FILES_DIRECTORY.resolve())
            allowed_directories = _parse_allowed_directories(allowed_report)
            if allowed_directories != (expected_directory,):
                raise FilesystemServerError(
                    "Filesystem server reported an unexpected directory allowlist. "
                    f"Expected only {expected_directory}; received "
                    f"{allowed_directories or 'no directories'}."
                )

            sample_content = _tool_text(
                await client.call_tool(
                    "read_text_file",
                    {"path": str(sample_path)},
                ),
                "read_text_file",
            )
    except FilesystemServerError:
        raise
    except (MCPError, OSError, TimeoutError, ExceptionGroup) as exc:
        raise FilesystemServerError(
            "Could not connect to the published Filesystem MCP server. "
            "Check that Node.js 24+ is installed and run `npm ci` in hw4."
        ) from exc

    return FilesystemVerification(
        tools=tools,
        allowed_directories=allowed_directories,
        sample_content=sample_content,
    )


async def _run_check() -> None:
    """Print the verified connection details and sample document."""
    result = await verify_filesystem_server()
    print("Discovered Filesystem tools:")
    for tool_name in result.tools:
        print(f"- {tool_name}")
    print("\nAllowed directories:")
    for directory in result.allowed_directories:
        print(directory)
    print("\nSample document:")
    print(result.sample_content)


def main() -> None:
    """Run the Filesystem server connection check from the command line."""
    try:
        asyncio.run(_run_check())
    except FilesystemServerError as exc:
        print(f"Filesystem server check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
