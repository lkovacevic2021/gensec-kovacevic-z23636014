# ThreatLens

ThreatLens is a Streamlit application that scans pasted text or `.txt`/`.log`
uploads for common patterns associated with SQL injection, cross-site scripting
(XSS), command injection, and path traversal.

## Run locally

From the repository root, create and activate a virtual environment, install
the homework dependencies, and start Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r hw6\requirements.txt
python -m streamlit run hw6\app.py
```

Then open the local URL printed by Streamlit, usually
`http://localhost:8501`.

The application also includes `hw6/.streamlit/config.toml`, which configures
Streamlit's server-side upload limit to 1 MiB. Streamlit reads this
configuration before loading the application, including when launched from the
repository root. The application also validates the raw file size as an
additional security measure.

Run the automated tests from the `hw6` directory:

```powershell
python -m unittest discover -s . -p "test_*.py"
```

## Safety notes

- Uploaded files are limited to 1 MiB and are decoded as UTF-8 with invalid
  bytes replaced safely. The extension is checked against the allowed `.txt`
  and `.log` set before scanning.
- Pasted text is limited to 100,000 characters, and oversized input is rejected
  before pattern matching.
- Uploaded and pasted content is treated only as data. ThreatLens never
  executes it and does not use `eval`, `exec`, or shell commands.
- Results are displayed as ordinary text/table values rather than rendered
  user-provided HTML.
- Detection is pattern-based and may produce false positives or miss attacks.
  A clean result is not proof that input is safe.
