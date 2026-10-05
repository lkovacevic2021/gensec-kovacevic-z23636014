"""Command-line foundation for the Homework 3 Cybersecurity Research Agent."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
MODEL_ENV_VAR = "GOOGLE_MODEL"
DEFAULT_VERTEX_LOCATION = "us-west1"
COMMAND_TIMEOUT_SECONDS = 10
MAX_COMMAND_OUTPUT_CHARACTERS = 4000


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


def run_chat() -> None:
	"""Run a simple interactive chat loop with the configured Gemini model."""
	load_dotenv(Path(__file__).with_name(".env"))
	try:
		model = create_chat_model()
	except RuntimeError as error:
		raise SystemExit(str(error)) from error

	print("Cybersecurity Research Agent model check. Type 'quit' or 'exit' to stop.")
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

		response = model.invoke(user_message)
		print(f"Gemini: {response.content}\n")


def main() -> None:
	"""Run the direct Terminal test or start the Gemini command-line chat."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--test-terminal",
		action="store_true",
		help="run a direct Python version check through the Terminal tool",
	)
	arguments = parser.parse_args()
	if arguments.test_terminal:
		test_terminal_tool()
	else:
		run_chat()


if __name__ == "__main__":
	main()
