"""Command-line foundation for the Homework 3 Cybersecurity Research Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
MODEL_ENV_VAR = "GOOGLE_MODEL"
DEFAULT_VERTEX_LOCATION = "us-west1"


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


if __name__ == "__main__":
	run_chat()
