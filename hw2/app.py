"""A small NotebookLM-like Chainlit application for the course RAG database.

The vector database must be created first by running ``07_rag_loaddb.py``.
This application only queries the persisted Chroma collection and does not
reload source documents every time the chat starts.

Required environment variables:
	GOOGLE_CLOUD_PROJECT: Google Cloud project used by Vertex AI embeddings.
	GOOGLE_MODEL: Chat model name accepted by ``ChatGoogleGenerativeAI``.

Google authentication is supplied through the normal Google Cloud or
Generative AI environment configuration and is intentionally not stored here.
"""

import os
from pathlib import Path

import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings


PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
MODEL_ENV_VAR = "GOOGLE_MODEL"
EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_VERTEX_LOCATION = "us-west1"
DATABASE_PATH = Path(__file__).resolve().parent / "rag_data" / ".chromadb"


def required_environment_variable(name: str) -> str:
	"""Return a required environment variable or raise a useful error."""
	value = os.getenv(name)
	if not value:
		raise RuntimeError(
			f"Set the {name} environment variable before starting Chainlit."
		)
	return value


def create_vectorstore() -> Chroma:
	"""Open the Chroma database created by the Lab 02.3 loader."""
	if not DATABASE_PATH.exists():
		raise FileNotFoundError(
			f"The RAG database was not found at {DATABASE_PATH}. "
			"Run 07_rag_loaddb.py first."
		)

	embeddings = VertexAIEmbeddings(
		model_name=EMBEDDING_MODEL,
		project=required_environment_variable(PROJECT_ENV_VAR),
		location=os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_VERTEX_LOCATION),
	)
	return Chroma(
		embedding_function=embeddings,
		persist_directory=str(DATABASE_PATH),
	)


def format_documents(documents: list[Document]) -> str:
	"""Combine retrieved document chunks into the prompt context."""
	return "\n\n".join(document.page_content for document in documents)


def source_names(documents: list[Document]) -> list[str]:
	"""Return unique source labels in the order they first appear."""
	sources = []
	for document in documents:
		source = (
			document.metadata.get("source")
			or document.metadata.get("file_path")
			or document.metadata.get("path")
			or "Unknown source"
		)
		if source not in sources:
			sources.append(source)
	return sources


vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model=required_environment_variable(MODEL_ENV_VAR))
prompt = ChatPromptTemplate.from_template(
	"""You are a helpful assistant for question-answering over a collection of documents.
Answer using only the retrieved context below. If the context does not contain
the answer, say that you do not know. Keep the answer concise and factual.

Question: {question}

Context:
{context}

Answer:"""
)
answer_chain = prompt | llm | StrOutputParser()


@cl.on_chat_start
async def on_chat_start() -> None:
	"""Welcome the user and show which sources are available for questions."""
	metadata = vectorstore.get().get("metadatas", [])
	sources = sorted(
		{item.get("source", "Unknown source") for item in metadata if item}
	)
	source_text = "\n".join(f"- {source}" for source in sources)
	welcome = "**Welcome to the RAG document assistant.**\n\n"
	welcome += "Ask a question about the indexed documents.\n\n"
	welcome += "**Indexed sources**\n" + (source_text or "No sources found.")
	await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
	"""Retrieve relevant chunks and send a concise grounded answer."""
	documents = retriever.invoke(message.content)
	if not documents:
		await cl.Message(content="I could not find relevant documents.").send()
		return

	answer = answer_chain.invoke(
		{
			"question": message.content,
			"context": format_documents(documents),
		}
	)
	sources = "\n".join(f"- {source}" for source in source_names(documents))
	await cl.Message(content=f"{answer}\n\n**Sources**\n{sources}").send()


if __name__ == "__main__":
	cl.run()
