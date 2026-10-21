"""Tests for note storage and the custom MCP tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcp import Client

from study_assistant.server import create_server
from study_assistant.storage import NoteStore


class NoteStoreTests(unittest.TestCase):
    """Verify SQLite persistence and input validation."""

    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            suffix=".sqlite3",
            dir=Path(__file__).resolve().parents[1],
            delete=False,
        )
        handle.close()
        self.database_path = Path(handle.name)
        self.addCleanup(self.database_path.unlink, missing_ok=True)
        self.store = NoteStore(self.database_path)

    def test_saves_and_searches_notes(self) -> None:
        saved = self.store.save_note(
            "Cell biology",
            "Mitochondria generate energy for the cell.",
        )

        results = self.store.search_notes("mitochondria")

        self.assertEqual(results[0]["id"], saved["id"])
        self.assertEqual(results[0]["title"], "Cell biology")
        note = self.store.get_note(saved["id"])
        self.assertIsNotNone(note)
        self.assertEqual(note["content"], "Mitochondria generate energy for the cell.")

    def test_rejects_empty_note_and_search_query(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save_note(" ", "Some content")
        with self.assertRaises(ValueError):
            self.store.search_notes(" ")


class MCPToolTests(unittest.IsolatedAsyncioTestCase):
    """Verify the registered tools can complete a study-note workflow."""

    async def test_tools_save_search_and_generate_questions(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            suffix=".sqlite3",
            dir=Path(__file__).resolve().parents[1],
            delete=False,
        )
        handle.close()
        database_path = Path(handle.name)
        self.addCleanup(database_path.unlink, missing_ok=True)

        async with Client(
            create_server(NoteStore(database_path)),
            raise_exceptions=True,
        ) as client:
            tools = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in tools.tools},
                {
                    "save_note",
                    "search_notes",
                    "generate_practice_questions",
                },
            )

            saved = await client.call_tool(
                "save_note",
                {
                    "title": "Cell biology",
                    "content": (
                        "The nucleus stores DNA and helps regulate gene expression. "
                        "Mitochondria convert energy from food into ATP through "
                        "cellular respiration."
                    ),
                },
            )
            note_id = saved.structured_content["id"]
            search = await client.call_tool(
                "search_notes",
                {"query": "mitochondria"},
            )
            questions = await client.call_tool(
                "generate_practice_questions",
                {"note_id": note_id, "count": 3},
            )

        self.assertEqual(search.structured_content["result"][0]["id"], note_id)
        generated = questions.structured_content["result"]
        self.assertEqual(len(generated), 3)
        self.assertEqual(
            len({item["question"] for item in generated}),
            3,
        )
        self.assertIn(
            "What does the nucleus store?",
            [item["question"] for item in generated],
        )
        self.assertIn(
            "What does the nucleus help regulate?",
            [item["question"] for item in generated],
        )
        self.assertNotIn(
            "What does the note say about",
            "\n".join(item["question"] for item in generated),
        )
