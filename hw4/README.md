# AI Study Assistant

A local-first study assistant with a Gemini-powered interactive CLI. Gemini
decides which MCP tools to call and can coordinate multiple steps between the
custom Python notes server and the official published Filesystem server.

The custom server saves/searches SQLite study notes and generates basic practice
questions. The Filesystem server is restricted to `study_files`; the agent only
exposes its read-only tools to Gemini.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 24 or newer and npm
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), for Vertex AI
  Application Default Credentials

## Setup

From this directory, install the Python and npm dependencies:

```powershell
uv sync
npm ci
```

The Filesystem server is pinned in `package.json` and `package-lock.json`. Its
only allowed directory is the absolute `hw4/study_files` path, computed by the
Python client at runtime. On Windows this is passed as a full path such as
`C:\path\to\hw4\study_files`.

## Configure Gemini on Vertex AI

Enable the Vertex AI API for a Google Cloud project, authenticate with ADC, and
configure the project and location in the current PowerShell session:

```powershell
gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT = "your-google-cloud-project-id"
$env:GOOGLE_CLOUD_LOCATION = "us-central1"
```

The agent uses ADC through the Google Gen AI SDK; it does not require or store an
API key. `GOOGLE_CLOUD_LOCATION` defaults to `us-central1`. The default model is
`gemini-2.5-flash`; override it with `STUDY_ASSISTANT_MODEL` if needed.

## Run the interactive AI agent

From `hw4`, start the CLI:

```powershell
uv run study-assistant-agent
```

The host starts both MCP servers over standard I/O, discovers their tools, and
prints the available tools. Enter natural-language requests at the `You:`
prompt. The model selects and sequences tool calls; the terminal shows each
server and tool as it runs. Type `quit` or `exit` to stop.

For example, try:

```text
Read study_files/cell-biology.md, summarize the key ideas in a few sentences,
save the summary as a note titled "Cell biology review", then generate three
practice questions from that saved note and show me the questions and answers.
```

Gemini can respond by reading the document through the Filesystem MCP server,
passing its content to `save_note` on the custom server, then calling
`generate_practice_questions` using the saved note ID. It can also use
`search_notes` for follow-up requests. The Filesystem tools exposed to Gemini
are read-only, and the server itself is restricted to `study_files`.

## Verify the Filesystem MCP connection

Run the verification command from `hw4`:

```powershell
uv run study-assistant-filesystem-check
```

The Python client launches the installed npm package over standard I/O,
discovers its tools, asks the server to report its allowed directories, and
reads `study_files/cell-biology.md` with `read_text_file`. The check prints the
discovered tools, server-reported allowlist, and sample document. It exits with
an error if the package cannot start, expected tools are missing, or the
directory is not reported as allowed.

The same connection check is covered by the automated tests below.

## Run the custom MCP server by itself

Start the server using its standard-I/O transport:

```powershell
uv run study-assistant
```

The process waits for an MCP client on standard input/output. To interactively
test its tools, connect an MCP client or the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector) using this
command and arguments:

```text
Command: uv
Arguments: --directory <absolute-path-to-hw4> run study-assistant
```

Try the tools in this order:

1. `save_note` with a title and content containing a few sentences.
2. `search_notes` with a word or phrase from that note.
3. `generate_practice_questions` with the returned note ID and a count from 1
   to 10.

Run the automated checks from `hw4` after `uv sync` and `npm ci`:

```powershell
uv run python -m unittest discover -s tests -v
```

The database is created as `study_notes.db` when the server starts. Delete that
file to remove locally saved notes.

## Available tools

- `save_note(title, content)` saves a note and returns its ID, title, and
  creation time.
- `search_notes(query, limit=10)` finds matching titles or content. The limit
  must be from 1 to 100.
- `generate_practice_questions(note_id, count=5)` creates recall questions and
  answers from sentences in an existing note. The count must be from 1 to 10.

Invalid input and database failures are reported as tool errors; the server does
not silently ignore them.
