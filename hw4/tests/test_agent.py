"""Mocked tests for model-directed MCP orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from google.genai import types
from mcp import Client

from study_assistant.agent import (
    StudyAssistantAgent,
    _filesystem_tool_arguments,
    _server_parameters,
)
from study_assistant.filesystem_client import STUDY_FILES_DIRECTORY


def tool_definition(name: str) -> SimpleNamespace:
    """Create a minimal MCP tool definition for an orchestration test."""
    return SimpleNamespace(
        name=name,
        description=f"Test tool {name}.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "title": "Path"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def function_response(name: str, arguments: dict[str, object]) -> object:
    """Wrap a model-selected function call in a Gemini candidate response."""
    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name=name, args=arguments)
            )
        ],
    )
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def text_response(text: str) -> object:
    """Wrap a final text answer in a Gemini candidate response."""
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class FakeMCPClient:
    """Minimal async MCP client fake with inspectable tool calls."""

    def __init__(self, tools: list[SimpleNamespace], result: dict[str, object]) -> None:
        self.tools = tools
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=self.tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> SimpleNamespace:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            structured_content=self.result,
            content=[],
        )


class FakeModel:
    """Return a predetermined model-driven sequence of tool calls."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> object:
        self.requests.append(
            {"model": model, "contents": contents, "config": config}
        )
        return self.responses.pop(0)


class StudyAssistantAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify Gemini's choices route across both MCP servers."""

    def test_filesystem_arguments_resolve_relative_and_absolute_root_prefixes(self) -> None:
        self.assertEqual(
            _filesystem_tool_arguments(
                "read_text_file",
                {"path": "study_files/study_files/cell-biology.md"},
            ),
            {"path": "cell-biology.md"},
        )
        self.assertEqual(
            _filesystem_tool_arguments(
                "read_text_file",
                {
                    "path": str(
                        STUDY_FILES_DIRECTORY
                        / "study_files"
                        / "cell-biology.md"
                    )
                },
            ),
            {"path": "cell-biology.md"},
        )
        self.assertEqual(
            _filesystem_tool_arguments(
                "read_multiple_files",
                {
                    "paths": [
                        "study_files/cell-biology.md",
                        "other-notes.md",
                    ]
                },
            ),
            {"paths": ["cell-biology.md", "other-notes.md"]},
        )

    async def test_model_coordinates_read_save_and_question_tools(self) -> None:
        filesystem = FakeMCPClient(
            [tool_definition("read_text_file"), tool_definition("write_file")],
            {"text": "Mitochondria convert food energy into ATP."},
        )
        notes = FakeMCPClient(
            [
                tool_definition("save_note"),
                tool_definition("generate_practice_questions"),
            ],
            {"id": 12, "title": "Cell energy"},
        )
        model = FakeModel(
            [
                function_response(
                    "read_text_file",
                    {"path": "study_files/study_files/cell-biology.md"},
                ),
                function_response(
                    "save_note",
                    {
                        "title": "Cell energy",
                        "content": "Mitochondria convert food energy into ATP.",
                    },
                ),
                function_response(
                    "generate_practice_questions",
                    {"note_id": 12, "count": 3},
                ),
                text_response("Saved the summary and generated three questions."),
            ]
        )
        printed: list[str] = []
        agent = StudyAssistantAgent(model, output=printed.append)

        answer = await agent.respond(
            "Read the cell biology file, save a summary, and make three questions.",
            [("Study Notes", notes), ("Filesystem", filesystem)],
        )

        self.assertEqual(
            answer,
            "Saved the summary and generated three questions.",
        )
        self.assertEqual(
            [name for name, _ in filesystem.calls],
            ["read_text_file"],
        )
        self.assertEqual(
            filesystem.calls[0][1]["path"],
            "cell-biology.md",
        )
        self.assertEqual(
            [name for name, _ in notes.calls],
            ["save_note", "generate_practice_questions"],
        )
        self.assertEqual(len(model.requests), 4)
        declarations = model.requests[0]["config"].tools[0].function_declarations
        declarations_by_name = {
            declaration.name: declaration for declaration in declarations
        }
        self.assertIn("read_text_file", declarations_by_name)
        self.assertNotIn(
            "write_file",
            declarations_by_name,
        )
        schema = declarations_by_name["read_text_file"].parameters.model_dump(
            exclude_none=True,
            by_alias=True,
        )
        self.assertNotIn("title", schema["properties"]["path"])
        self.assertIn(
            "relative to the allowed study_files root",
            declarations_by_name["read_text_file"].description,
        )
        self.assertIn(
            'never prefix paths with "study_files/"',
            model.requests[0]["config"].system_instruction,
        )
        self.assertEqual(
            sum("[Filesystem MCP]" in line or "[Study Notes MCP]" in line for line in printed),
            3,
        )

    async def test_empty_request_is_rejected(self) -> None:
        agent = StudyAssistantAgent(FakeModel([]))

        with self.assertRaises(ValueError):
            await agent.respond("  ", [])

    async def test_duplicate_tool_names_are_reported(self) -> None:
        model = FakeModel([])
        agent = StudyAssistantAgent(model)
        client = FakeMCPClient([tool_definition("search_notes")], {})

        with self.assertRaisesRegex(RuntimeError, "provided by both"):
            await agent.discover_tools(
                [("Study Notes", client), ("Other MCP", client)]
            )

    async def test_custom_server_launches_as_an_mcp_subprocess(self) -> None:
        custom_server, _ = _server_parameters()
        database_path = Path(__file__).resolve().parents[1] / "study_notes.db"
        database_existed = database_path.exists()

        try:
            async with Client(custom_server, raise_exceptions=True) as client:
                tools = await client.list_tools()
        finally:
            if not database_existed:
                database_path.unlink(missing_ok=True)

        self.assertEqual(
            {tool.name for tool in tools.tools},
            {
                "save_note",
                "search_notes",
                "generate_practice_questions",
            },
        )
