"""Conexion SQLite compartida por FTS5 y sqlite-vec (mismo archivo .db).

sqlite3 es sincrono: toda operacion se invoca desde `asyncio.to_thread`. La
conexion se abre con check_same_thread=False y un lock serializa el acceso,
porque los hilos del pool varian entre llamadas.
"""

import sqlite3
import threading
from pathlib import Path

import sqlite_vec

EMBEDDING_DIM = 384  # intfloat/multilingual-e5-small

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def init(sqlite_path: str) -> None:
    """Abre la conexion, carga sqlite-vec y crea el esquema. Idempotente."""
    global _conn
    with _lock:
        if _conn is not None:
            return

        path = Path(sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Tabla real: da PRIMARY KEY para idempotencia via INSERT OR IGNORE.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs(
                id       TEXT PRIMARY KEY,
                content  TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        # Indice FTS5 de contenido externo sobre docs, sincronizado por triggers.
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                content,
                content='docs',
                content_rowid='rowid'
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, content) VALUES (new.rowid, new.content);
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END
            """
        )
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(
                doc_id    TEXT PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            )
            """
        )
        conn.commit()
        _conn = conn


def execute(sql: str, params: tuple = ()) -> list[tuple]:
    """Ejecuta una sentencia y devuelve todas las filas."""
    with _lock:
        if _conn is None:
            raise RuntimeError("SQLite no inicializado: llamar a db.init() primero")
        cur = _conn.execute(sql, params)
        rows = cur.fetchall()
        _conn.commit()
        return rows


def executemany(sql: str, seq: list[tuple]) -> int:
    """Ejecuta una sentencia por cada tupla. Devuelve filas afectadas."""
    with _lock:
        if _conn is None:
            raise RuntimeError("SQLite no inicializado: llamar a db.init() primero")
        cur = _conn.executemany(sql, seq)
        _conn.commit()
        return cur.rowcount


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
