import sqlite3
from pathlib import Path

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"
TOP_K = 3

ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "db" / "rag.db"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided below. Do not use any prior knowledge or make assumptions beyond "
    "what is stated in the context.\n"
    "If the context does not contain the information needed to answer the "
    "question, respond exactly with: \"I don't know based on the provided "
    "context.\" Never guess or fabricate an answer."
)

_embedding_client = None
_chat_client = None


def _get_manager() -> FoundryLocalManager:
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="local-rag-app"))
    return FoundryLocalManager.instance


def _load_model(model_alias: str):
    manager = _get_manager()
    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise RuntimeError(f"Model alias '{model_alias}' not found in the Foundry Local catalog.")
    model.download()
    model.load()
    return model


def _get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = _load_model(EMBEDDING_MODEL_ALIAS).get_embedding_client()
    return _embedding_client


def _get_chat_client():
    global _chat_client
    if _chat_client is None:
        _chat_client = _load_model(CHAT_MODEL_ALIAS).get_chat_client()
    return _chat_client


def get_chat_client():
    """Public accessor for the Phi-3.5-mini chat client, for reuse outside the RAG pipeline."""
    return _get_chat_client()


def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1e-10
    return (matrix @ query_vector) / denom


def get_top_chunks(query: str, top_k: int = TOP_K) -> list[str]:
    """Embed `query` and return the top_k stored chunks ranked by cosine similarity."""
    query_response = _get_embedding_client().generate_embedding(query)
    query_vector = np.array(query_response.data[0].embedding, dtype=np.float32)

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT content, embedding FROM documents").fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    contents = [content for content, _ in rows]
    matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])

    similarities = _cosine_similarity(query_vector, matrix)
    ranked_indices = np.argsort(-similarities)[:top_k]

    return [contents[i] for i in ranked_indices]


def answer_query(user_question: str) -> str:
    """Retrieve relevant context for `user_question` and generate a grounded answer."""
    top_chunks = get_top_chunks(user_question)
    context = "\n\n---\n\n".join(top_chunks) if top_chunks else "(no documents available)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"},
    ]

    completion = _get_chat_client().complete_chat(messages)
    return completion.choices[0].message.content
