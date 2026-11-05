import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    MAX_FILE_SIZE_BYTES,
    MAX_PASTED_TEXT_CHARS,
    read_uploaded_file,
    scan_text,
)


class UploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class ThreatLensTests(unittest.TestCase):
    def test_streamlit_upload_limit_is_one_mib(self):
        config = Path(__file__).parent / ".streamlit" / "config.toml"
        self.assertIn("maxUploadSize = 1", config.read_text(encoding="utf-8"))

    def test_detects_sql_injection(self):
        self.assertTrue(any(f.category == "SQL injection" for f in scan_text("' OR '1'='1' --")))

    def test_detects_xss(self):
        self.assertTrue(any(f.category == "Cross-site scripting (XSS)" for f in scan_text("<script>alert(1)</script>")))

    def test_detects_command_injection(self):
        self.assertTrue(any(f.category == "Command injection" for f in scan_text("report.txt; whoami")))

    def test_detects_path_traversal(self):
        self.assertTrue(any(f.category == "Path traversal" for f in scan_text("../../etc/passwd")))

    def test_normal_text_has_no_findings(self):
        self.assertEqual(scan_text("The quarterly report is ready."), [])

    def test_oversized_pasted_input_is_rejected(self):
        with self.assertRaises(ValueError):
            scan_text("x" * (MAX_PASTED_TEXT_CHARS + 1))

    def test_oversized_input_is_rejected_before_decoding(self):
        uploaded = UploadedFile("large.txt", b"x" * (MAX_FILE_SIZE_BYTES + 1))
        with patch("app.st.error") as error:
            self.assertIsNone(read_uploaded_file(uploaded))
        error.assert_called_once()

    def test_invalid_file_content_is_replaced_safely(self):
        uploaded = UploadedFile("input.txt", b"safe \xff text")
        with patch("app.st.warning"):
            result = read_uploaded_file(uploaded)
        self.assertEqual(result, "safe \ufffd text")

    def test_invalid_extension_is_rejected(self):
        uploaded = UploadedFile("input.html", b"<script>alert(1)</script>")
        with patch("app.st.error") as error:
            self.assertIsNone(read_uploaded_file(uploaded))
        error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
