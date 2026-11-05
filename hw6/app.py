"""ThreatLens: a pattern-based security input analyzer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import streamlit as st


MAX_FILE_SIZE_BYTES = 1_048_576
MAX_PASTED_TEXT_CHARS = 100_000
ALLOWED_EXTENSIONS = frozenset({".txt", ".log"})


@dataclass(frozen=True)
class DetectionRule:
    category: str
    severity: str
    explanation: str
    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    explanation: str
    rule_name: str
    line: int


def compile_rule(
    category: str,
    severity: str,
    explanation: str,
    name: str,
    expression: str,
) -> DetectionRule:
    return DetectionRule(
        category=category,
        severity=severity,
        explanation=explanation,
        name=name,
        pattern=re.compile(expression, re.IGNORECASE | re.MULTILINE),
    )


RULES = (
    compile_rule(
        "SQL injection",
        "High",
        "SQL syntax or comment markers may be attempting to alter a database query.",
        "SQL operator/comment sequence",
        r"""(?:'\s*(?:OR|AND)\s+['"\w\d]+\s*=\s*['"\w\d]+|(?:UNION\s+ALL\s+SELECT|UNION\s+SELECT)\b|(?:--|/\*|\*/|#)\s*(?:$|\S))""",
    ),
    compile_rule(
        "SQL injection",
        "High",
        "A database-specific statement or delay function was detected.",
        "SQL statement/function",
        r"""\b(?:DROP\s+TABLE|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|SLEEP\s*\(|BENCHMARK\s*\(|WAITFOR\s+DELAY)\b""",
    ),
    compile_rule(
        "Cross-site scripting (XSS)",
        "High",
        "Markup or a script-capable event handler may execute code in a browser.",
        "Script or event-handler markup",
        r"""<\s*(?:script|iframe|object|embed)\b[^>]*>|on(?:error|load|click|mouseover)\s*=\s*["']?[^ >]+|javascript\s*:""",
    ),
    compile_rule(
        "Command injection",
        "High",
        "Shell control syntax or a command substitution may cause unintended command execution.",
        "Shell metacharacter or substitution",
        r"""(?:[;&|]\s*(?:\w+|/)|`[^`\r\n]+`|\$\([^)\r\n]+\)|\b(?:cmd(?:\.exe)?|powershell(?:\.exe)?|bash|sh|zsh)\s+-c\b)""",
    ),
    compile_rule(
        "Path traversal",
        "Medium",
        "Parent-directory navigation may allow access outside the intended directory.",
        "Parent-directory sequence",
        r"""(?:\.\.[\\/]){1,}|(?:%2e%2e|%252e%252e)(?:%2f|%5c|[\\/])""",
    ),
)

EXAMPLES = {
    "SQL injection": "' OR '1'='1' --",
    "Cross-site scripting (XSS)": '<script>alert("test")</script>',
    "Command injection": "report.txt; whoami",
    "Path traversal": "../../etc/passwd",
    "Benign example": "Please review the quarterly access report.",
}


def scan_text(text: str) -> list[Finding]:
    """Return one finding for every rule match without executing the input."""
    if len(text) > MAX_PASTED_TEXT_CHARS:
        raise ValueError(
            f"Input exceeds the {MAX_PASTED_TEXT_CHARS:,}-character limit."
        )

    findings: list[Finding] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(
                Finding(
                    category=rule.category,
                    severity=rule.severity,
                    explanation=rule.explanation,
                    rule_name=rule.name,
                    line=line,
                )
            )
    return sorted(findings, key=lambda finding: (finding.line, finding.category))


def read_uploaded_file(uploaded_file: Any) -> str | None:
    """Validate and read a bounded UTF-8 text file without executing it."""
    filename = getattr(uploaded_file, "name", "")
    if not filename or PurePath(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        st.error("Only .txt and .log files are accepted.")
        return None

    raw_data = uploaded_file.getvalue()
    if len(raw_data) > MAX_FILE_SIZE_BYTES:
        st.error("That file is larger than the 1 MiB upload limit.")
        return None

    try:
        return raw_data.decode("utf-8")
    except UnicodeDecodeError:
        st.warning("The file is not valid UTF-8. Invalid bytes were replaced before scanning.")
        return raw_data.decode("utf-8", errors="replace")


def display_findings(findings: list[Finding]) -> None:
    st.subheader("Findings")
    if not findings:
        st.success("No suspicious patterns were found in the supplied input.")
        return

    st.warning(
        f"Found {len(findings)} suspicious pattern"
        f"{'s' if len(findings) != 1 else ''}."
    )
    st.dataframe(
        [
            {
                "Threat category": finding.category,
                "Severity": finding.severity,
                "Explanation": finding.explanation,
                "Rule": finding.rule_name,
                "Line": finding.line,
            }
            for finding in findings
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="ThreatLens", page_icon="🛡️")
    st.title("ThreatLens")
    st.caption("A security input analyzer for common suspicious patterns")
    st.info(
        "ThreatLens uses pattern matching only. It can produce false positives and "
        "miss attacks that use an unexpected encoding or a pattern not covered by "
        "these rules. Do not treat a clean result as proof that input is safe."
    )

    with st.sidebar:
        st.header("Examples")
        selected_example = st.selectbox("Choose an example", tuple(EXAMPLES))
        if st.button("Use selected example"):
            st.session_state["threatlens_input"] = EXAMPLES[selected_example]
        st.caption("Upload limit: 1 MiB. Accepted extensions: .txt and .log.")

    uploaded_file = st.file_uploader(
        "Upload text to scan",
        type=["txt", "log"],
        help="Files are read as UTF-8 and are never executed.",
    )
    pasted_text = st.text_area(
        "Or paste text to scan",
        key="threatlens_input",
        height=180,
        placeholder="Paste an input value, request parameter, or log line here...",
    )

    input_text = pasted_text
    if len(input_text) > MAX_PASTED_TEXT_CHARS:
        st.error(
            f"Pasted input exceeds the {MAX_PASTED_TEXT_CHARS:,}-character limit."
        )
        return

    if uploaded_file is not None:
        uploaded_text = read_uploaded_file(uploaded_file)
        if uploaded_text is None:
            return
        input_text = uploaded_text
        st.caption(f"Scanning uploaded file: {uploaded_file.name}")

    if st.button("Scan input", type="primary"):
        if not input_text.strip():
            st.warning("Enter text or upload a file before scanning.")
        else:
            display_findings(scan_text(input_text))


if __name__ == "__main__":
    main()