"""MCP tools for saving, searching, and studying notes."""

from __future__ import annotations

import re

from mcp.server.mcpserver import MCPServer

from study_assistant.storage import NoteStore

_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "first",
    "from",
    "have",
    "into",
    "other",
    "should",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "using",
    "which",
    "while",
    "with",
}


def create_server(note_store: NoteStore | None = None) -> MCPServer:
    """Build the custom MCP server, optionally using a supplied note store."""
    store = note_store or NoteStore()
    server = MCPServer(
        name="study-notes",
        instructions=(
            "Save and search local study notes, and generate basic recall questions "
            "from a saved note."
        ),
    )

    @server.tool()
    def save_note(title: str, content: str) -> dict[str, int | str]:
        """Save a study note with a short title and its content."""
        return store.save_note(title, content)

    @server.tool()
    def search_notes(query: str, limit: int = 10) -> list[dict[str, int | str]]:
        """Search saved note titles and contents for a word or phrase."""
        return store.search_notes(query, limit)

    @server.tool()
    def generate_practice_questions(
        note_id: int, count: int = 5
    ) -> list[dict[str, str]]:
        """Create basic recall questions and answers from a saved note."""
        if not 1 <= count <= 10:
            raise ValueError("Question count must be between 1 and 10.")

        note = store.get_note(note_id)
        if note is None:
            raise ValueError(f"No study note exists with ID {note_id}.")

        sentences = [
            sentence.strip(" \t\r\n-*")
            for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", note["content"])
            if sentence.strip(" \t\r\n-*")
        ]
        if not sentences:
            raise ValueError(f"Study note {note_id} has no usable content.")

        questions = []
        for sentence in sentences[:count]:
            keywords = [
                word
                for word in _WORD_PATTERN.findall(sentence)
                if len(word) > 4 and word.casefold() not in _STOP_WORDS
            ]
            topic = keywords[0] if keywords else str(note["title"])
            questions.append(
                {
                    "question": f"What does the note say about {topic}?",
                    "answer": sentence,
                }
            )
        return questions

    return server


def main() -> None:
    """Run the study-notes MCP server over standard input and output."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
