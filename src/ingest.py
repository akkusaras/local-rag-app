import sqlite3
from pathlib import Path

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

MODEL_ALIAS = "qwen3-embedding-0.6b"
PARAGRAPHS_PER_CHUNK = 2  # groups paragraphs into chunks of 1-3

ROOT_DIR = Path(__file__).parent.parent
SAMPLE_FILE = ROOT_DIR / "data" / "sample.txt"
DB_PATH = ROOT_DIR / "db" / "rag.db"


def load_paragraphs(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_paragraphs(paragraphs: list[str], size: int = PARAGRAPHS_PER_CHUNK) -> list[str]:
    return [
        "\n\n".join(paragraphs[i:i + size])
        for i in range(0, len(paragraphs), size)
    ]


def get_embedding_client(model_alias: str):
    FoundryLocalManager.initialize(Configuration(app_name="local-rag-app"))
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise RuntimeError(f"Model alias '{model_alias}' not found in the Foundry Local catalog.")

    print(f"Preparing model '{model.id}' (this downloads it on first run)...")
    model.download()
    model.load()

    return model.get_embedding_client()


def embed_chunks(client, chunks: list[str]) -> list[np.ndarray]:
    response = client.generate_embeddings(chunks)
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def store_chunks(db_path: Path, chunks: list[str], embeddings: list[np.ndarray]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)",
            [(chunk, embedding.tobytes()) for chunk, embedding in zip(chunks, embeddings)],
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE_FILE)
    chunks = chunk_paragraphs(paragraphs)
    print(f"Loaded {len(paragraphs)} paragraphs, split into {len(chunks)} chunks.")

    client = get_embedding_client(MODEL_ALIAS)
    embeddings = embed_chunks(client, chunks)

    store_chunks(DB_PATH, chunks, embeddings)
    print(f"Inserted {len(chunks)} chunks into {DB_PATH}")


if __name__ == "__main__":
    main()
