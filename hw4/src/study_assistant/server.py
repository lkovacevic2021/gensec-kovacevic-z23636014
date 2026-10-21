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
_PREDICATE_PATTERN = re.compile(
    r"\b(?P<verb>"
    r"helps?\s+regulate|"
    r"breaks?\s+down|"
    r"consists?\s+of|"
    r"stores?|contains?|holds?|regulates?|controls?|converts?|"
    r"generates?|produces?|assembles?|builds?|uses?|absorbs?|"
    r"carries?|forms?|supports?|protects?|releases?|transports?|"
    r"provides?|requires?|causes?|prevents?|"
    r"store|contain|hold|regulate|control|convert|generate|produce|"
    r"assemble|build|use|absorb|carry|form|support|protect|release|"
    r"transport|provide|require|cause|prevent"
    r")\b",
    re.IGNORECASE,
)
_VERB_BASE_FORMS = {
    "helps regulate": "help regulate",
    "help regulate": "help regulate",
    "breaks down": "break down",
    "break down": "break down",
    "consists of": "consist of",
    "consist of": "consist of",
    "stores": "store",
    "contains": "contain",
    "holds": "hold",
    "regulates": "regulate",
    "controls": "control",
    "converts": "convert",
    "generates": "generate",
    "produces": "produce",
    "assembles": "assemble",
    "builds": "build",
    "uses": "use",
    "absorbs": "absorb",
    "carries": "carry",
    "forms": "form",
    "supports": "support",
    "protects": "protect",
    "releases": "release",
    "transports": "transport",
    "provides": "provide",
    "requires": "require",
    "causes": "cause",
    "prevents": "prevent",
}
_CONJUNCTION_PATTERN = re.compile(r"\b(?:and|but|while)\b", re.IGNORECASE)
_DETAIL_PATTERN = re.compile(
    r"\b(?P<preposition>from|into|through|by|using|via|in|on|at)\s+"
    r"(?P<detail>.+?)(?=\s+\b(?:from|into|through|by|using|via|in|on|at)\b"
    r"|[,;.!?]|$)",
    re.IGNORECASE,
)


def _question_for_predicate(
    subject: str,
    verb: str,
    object_text: str,
) -> tuple[str, str]:
    """Create a direct recall question for a subject-predicate fact."""
    subject = re.sub(
        r"^(The|A|An)\b",
        lambda match: match.group().lower(),
        subject,
    )
    base_verb = _VERB_BASE_FORMS.get(verb.casefold(), verb.casefold())
    auxiliary = "does" if verb.casefold().split()[0].endswith("s") else "do"
    return (
        f"What {auxiliary} {subject} {base_verb}?",
        object_text.strip(" \t\r\n-*"),
    )


def _questions_for_sentence(sentence: str) -> list[dict[str, str]]:
    """Extract distinct subject-predicate and supporting-detail questions."""
    matches = list(_PREDICATE_PATTERN.finditer(sentence))
    if not matches:
        words = [
            word
            for word in _WORD_PATTERN.findall(sentence)
            if len(word) > 4 and word.casefold() not in _STOP_WORDS
        ]
        if not words:
            return []
        topic = words[0]
        return [
            {
                "question": f"What is an important idea about {topic}?",
                "answer": sentence.strip(" \t\r\n-*"),
            }
        ]

    root_subject = sentence[: matches[0].start()].strip(" \t\r\n-*")
    if not root_subject:
        return []

    questions: list[dict[str, str]] = []
    previous_match = None
    for index, match in enumerate(matches):
        verb = match.group("verb")
        if previous_match is None:
            subject = root_subject
        else:
            between = sentence[previous_match.end() : match.start()]
            conjunction = list(_CONJUNCTION_PATTERN.finditer(between))
            trailing_subject = (
                between[conjunction[-1].end() :].strip(" \t\r\n-*")
                if conjunction
                else ""
            )
            if trailing_subject and not trailing_subject.casefold().startswith(
                ("help ", "helps ", "break ", "breaks ", "consist ", "consists ")
            ):
                subject = trailing_subject
            else:
                subject = root_subject

        next_match = matches[index + 1] if index + 1 < len(matches) else None
        object_end = next_match.start() if next_match is not None else len(sentence)
        object_text = sentence[match.end() : object_end].strip(" \t\r\n-*")
        object_text = re.sub(
            r"\s+(?:and|but|while)\s*$",
            "",
            object_text,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n-*.,;:")
        if not object_text:
            previous_match = match
            continue

        primary_object = re.split(
            r"\b(?:through|by|using|via)\b",
            object_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" \t\r\n-*.,;:")
        question, answer = _question_for_predicate(
            subject,
            verb,
            primary_object or object_text,
        )
        questions.append({"question": question, "answer": answer})

        base_verb = _VERB_BASE_FORMS.get(verb.casefold(), verb.casefold())
        auxiliary = "does" if verb.casefold().split()[0].endswith("s") else "do"
        for detail in _DETAIL_PATTERN.finditer(object_text):
            preposition = detail.group("preposition").casefold()
            detail_text = detail.group("detail").strip(" \t\r\n-*.,;:")
            prefix = object_text[: detail.start()].strip(" \t\r\n-*.,;:")
            if not detail_text or not prefix:
                continue
            if preposition == "from":
                detail_question = (
                    f"From what {auxiliary} {subject} {base_verb} {prefix}?"
                )
            elif preposition == "into":
                detail_question = (
                    f"Into what {auxiliary} {subject} {base_verb} {prefix}?"
                )
            elif preposition in {"through", "via"}:
                detail_question = (
                    f"Through what process {auxiliary} {subject} "
                    f"{base_verb} {prefix}?"
                )
            elif preposition in {"by", "using"}:
                detail_question = (
                    f"How {auxiliary} {subject} {base_verb} {prefix}?"
                )
            else:
                detail_question = (
                    f"Where {auxiliary} {subject} {base_verb} {prefix}?"
                )
            questions.append(
                {"question": detail_question, "answer": detail_text}
            )
        previous_match = match
    return questions


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

        candidates = [
            question
            for sentence in sentences
            for question in _questions_for_sentence(sentence)
        ]
        unique_questions: list[dict[str, str]] = []
        seen_questions: set[str] = set()
        for candidate in candidates:
            normalized_question = candidate["question"].casefold()
            if normalized_question in seen_questions:
                continue
            seen_questions.add(normalized_question)
            unique_questions.append(candidate)
            if len(unique_questions) == count:
                break
        return unique_questions

    return server


def main() -> None:
    """Run the study-notes MCP server over standard input and output."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
