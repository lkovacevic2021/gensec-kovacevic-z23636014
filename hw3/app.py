"""Command-line Cybersecurity Research Agent for Homework 3."""

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
MODEL_ENV_VAR = "GOOGLE_MODEL"
DEFAULT_VERTEX_LOCATION = "us-west1"
COMMAND_TIMEOUT_SECONDS = 10
MAX_COMMAND_OUTPUT_CHARACTERS = 4000
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_REQUEST_TIMEOUT_SECONDS = 10
CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
CYBERSECURITY_SYSTEM_PROMPT = (
	"You are a concise cybersecurity research assistant. Use lookup_cve to "
	"retrieve authoritative NVD details for specific CVE IDs. Use terminal only "
	"for simple local environment diagnostics, such as checking software versions. "
	"This is a Windows environment; use 'python' (not 'python3') for Python commands. "
	"Answer general questions directly without tools when they are not needed. "
	"Distinguish verified tool results from general knowledge, and never invent "
	"tool output or CVE details."
)


def required_environment_variable(name: str) -> str:
	"""Return a required environment variable or raise a helpful error."""
	value = os.getenv(name)
	if not value:
		raise RuntimeError(
			f"Missing required environment variable {name}. "
			"Set it in hw3/.env or in your shell environment. "
			"Vertex AI authentication uses Application Default Credentials."
		)
	return value


def create_chat_model() -> ChatGoogleGenerativeAI:
	"""Create a Gemini model using Vertex AI and Application Default Credentials."""
	return ChatGoogleGenerativeAI(
		model=required_environment_variable(MODEL_ENV_VAR),
		project=required_environment_variable(PROJECT_ENV_VAR),
		location=os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_VERTEX_LOCATION),
		vertexai=True,
	)


@tool(
	"terminal",
	description=(
		"Run one local terminal command to inspect the Homework 3 environment. "
		"Use for simple diagnostics such as checking Python or Git versions. "
		"This is Windows: use 'python' rather than 'python3' for Python commands. "
		"Commands run in hw3 without a shell; pipes and shell operators are not supported."
	),
)
def terminal_tool(command: str) -> str:
	"""Execute one local command with a timeout and bounded captured output."""
	try:
		arguments = shlex.split(command)
		if not arguments:
			return "No command was provided."
		result = subprocess.run(
			arguments,
			cwd=Path(__file__).resolve().parent,
			capture_output=True,
			text=True,
			timeout=COMMAND_TIMEOUT_SECONDS,
			check=False,
		)
	except ValueError as error:
		return f"Could not parse command: {error}"
	except FileNotFoundError:
		return f"Command not found: {arguments[0]}"
	except subprocess.TimeoutExpired:
		return f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."

	output = "\n".join(
		part.strip()
		for part in (result.stdout, result.stderr)
		if part and part.strip()
	)
	if not output:
		output = "Command completed with no output."
	output += f"\nExit code: {result.returncode}"
	if len(output) > MAX_COMMAND_OUTPUT_CHARACTERS:
		output = output[:MAX_COMMAND_OUTPUT_CHARACTERS] + "\n[Output truncated.]"
	return output


def test_terminal_tool() -> None:
	"""Run a harmless direct check of the Terminal tool without Gemini."""
	print(terminal_tool.invoke({"command": "python --version"}))


@tool(
	"lookup_cve",
	description=(
		"Look up a CVE identifier in the NVD vulnerability database. "
		"Use this when asked for details or severity for a specific CVE ID, "
		"such as CVE-2021-44228."
	),
)
def cve_lookup_tool(cve_id: str) -> str:
	"""Retrieve and summarize a CVE record from the NVD CVE 2.0 API."""
	canonical_id = cve_id.strip().upper()
	if not CVE_ID_PATTERN.fullmatch(canonical_id):
		return "Invalid CVE ID format. Expected an ID such as CVE-2021-44228."

	headers = {}
	api_key = os.getenv("NVD_API_KEY")
	if api_key:
		headers["apiKey"] = api_key

	try:
		response = httpx.get(
			NVD_API_URL,
			params={"cveId": canonical_id},
			headers=headers,
			timeout=NVD_REQUEST_TIMEOUT_SECONDS,
		)
		response.raise_for_status()
		payload = response.json()
	except httpx.TimeoutException:
		return "The NVD request timed out. Please try again later."
	except httpx.HTTPStatusError as error:
		return f"The NVD API returned HTTP {error.response.status_code}."
	except httpx.RequestError as error:
		return f"Could not reach the NVD API: {error}."
	except ValueError:
		return "The NVD API returned invalid JSON."

	if not isinstance(payload, dict):
		return "The NVD API returned an unexpected response format."
	vulnerabilities = payload.get("vulnerabilities")
	if not isinstance(vulnerabilities, list):
		return "The NVD API response is missing vulnerability results."
	if not vulnerabilities:
		return (
			f"No NVD record was found for {canonical_id}.\n"
			f"NVD: https://nvd.nist.gov/vuln/detail/{canonical_id}"
		)

	cve_record = vulnerabilities[0].get("cve") if isinstance(vulnerabilities[0], dict) else None
	if not isinstance(cve_record, dict):
		return "The NVD API returned a malformed CVE record."

	description = "Description unavailable."
	descriptions = cve_record.get("descriptions")
	if isinstance(descriptions, list):
		for item in descriptions:
			if isinstance(item, dict) and item.get("lang") == "en" and item.get("value"):
				description = " ".join(str(item["value"]).split())
				break
	if len(description) > 1000:
		description = description[:997].rstrip() + "..."

	return "\n".join(
		[
			f"CVE: {canonical_id}",
			f"Description: {description}",
			f"CVSS: {_format_cvss(cve_record.get('metrics'))}",
			f"NVD: https://nvd.nist.gov/vuln/detail/{canonical_id}",
		]
	)


def _format_cvss(metrics: object) -> str:
	"""Return the first available CVSS score and severity from NVD metrics."""
	if not isinstance(metrics, dict):
		return "Not available"
	for metric_name in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
		metric_list = metrics.get(metric_name)
		if not isinstance(metric_list, list):
			continue
		for metric in metric_list:
			if not isinstance(metric, dict):
				continue
			cvss_data = metric.get("cvssData")
			if not isinstance(cvss_data, dict):
				continue
			score = cvss_data.get("baseScore")
			severity = cvss_data.get("baseSeverity") or metric.get("baseSeverity")
			if score is not None or severity:
				version = cvss_data.get("version") or metric_name.removeprefix("cvssMetricV")
				result = f"v{version}: {score}" if score is not None else f"v{version}"
				if severity:
					result += f" ({severity})"
				return result
	return "Not available"


def test_cve_tool(cve_id: str) -> None:
	"""Run a direct NVD lookup without creating the Gemini model."""
	load_dotenv(Path(__file__).with_name(".env"))
	print(cve_lookup_tool.invoke({"cve_id": cve_id}))


def create_research_agent(model: ChatGoogleGenerativeAI):
	"""Create the tool-calling cybersecurity agent around the Gemini model."""
	return create_agent(
		model=model,
		tools=[terminal_tool, cve_lookup_tool],
		system_prompt=CYBERSECURITY_SYSTEM_PROMPT,
	)


def human_readable_text(content: object) -> str:
	"""Extract displayable text from plain or structured model content."""
	if isinstance(content, str):
		return content.strip()
	if isinstance(content, dict):
		content_blocks = [content]
	elif isinstance(content, list):
		content_blocks = content
	else:
		return ""

	text_parts = []
	for block in content_blocks:
		if isinstance(block, str):
			text = block
		elif isinstance(block, dict):
			text = block.get("text", "")
		else:
			continue
		if isinstance(text, str) and text.strip():
			text_parts.append(text.strip())
	return "\n".join(text_parts)


def run_chat() -> None:
	"""Run an interactive chat loop with the tool-calling research agent."""
	load_dotenv(Path(__file__).with_name(".env"))
	try:
		model = create_chat_model()
		agent = create_research_agent(model)
	except RuntimeError as error:
		raise SystemExit(str(error)) from error

	print("Cybersecurity Research Agent. Type 'quit' or 'exit' to stop.")
	conversation = []
	while True:
		try:
			user_message = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nGoodbye.")
			break

		if user_message.lower() in {"quit", "exit"}:
			print("Goodbye.")
			break
		if not user_message:
			continue

		conversation.append({"role": "user", "content": user_message})
		try:
			result = agent.invoke({"messages": conversation})
		except Exception as error:
			conversation.pop()
			print(f"Agent error: {error}\nPlease try again.\n")
			continue

		conversation = result.get("messages", conversation)
		assistant_messages = [
			message
			for message in conversation
			if getattr(message, "type", None) == "ai"
		]
		if assistant_messages:
			answer = human_readable_text(assistant_messages[-1].content)
			if answer:
				print(f"Agent: {answer}\n")
			else:
				print("Agent returned no text response. Please try again.\n")
		else:
			print("Agent returned no response. Please try again.\n")


def main() -> None:
	"""Run a direct tool test or start the Gemini command-line chat."""
	parser = argparse.ArgumentParser(description=__doc__)
	test_options = parser.add_mutually_exclusive_group()
	test_options.add_argument(
		"--test-terminal",
		action="store_true",
		help="run a direct Python version check through the Terminal tool",
	)
	test_options.add_argument(
		"--test-cve",
		metavar="CVE_ID",
		help="look up a CVE ID directly using the NVD API",
	)
	arguments = parser.parse_args()
	if arguments.test_terminal:
		test_terminal_tool()
	elif arguments.test_cve:
		test_cve_tool(arguments.test_cve)
	else:
		run_chat()


if __name__ == "__main__":
	main()
