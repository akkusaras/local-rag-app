import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db" / "rag.db"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Initialized database at {db_path}")


if __name__ == "__main__":
    init_db()
