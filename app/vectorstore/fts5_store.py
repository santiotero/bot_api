"""Busqueda lexical y almacen del texto original (SQLite FTS5)."""

import asyncio
import json
import re

from app.vectorstore import db

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Los tokens se unen con OR, asi que una palabra funcional haria matchear casi
# cualquier documento y 'sin resultados' nunca se daria. Filtrarlas es lo que
# permite que la busqueda lexical discrimine.
_VACIAS = frozenset(
    """
    el la los las un una unos unas lo al del de en por para con sin sobre entre
    y o u ni que qué cual cuál cuales cuáles como cómo cuando cuándo donde dónde
    cuanto cuánto cuanta cuánta cuantos cuántos cuantas cuántas quien quién
    es son era eran ser soy sos somos estoy esta está estan están estar
    hay he ha han tengo tiene tienen tener hacer hace hacen
    me te se nos les le lo mi mis tu tus su sus
    si sí no mas más pero porque pues ya muy tan tanto
    este esta esto estos estas ese esa eso esos esas aquel aquella
    yo vos usted ustedes ellos ellas nosotros
    hola buenas buenos dias días tardes noches gracias favor porfa
    """.split()
)


def _fts5_query(texto: str) -> str | None:
    """Convierte texto libre en una expresion MATCH segura.

    El texto del usuario puede traer comillas, asteriscos u operadores de FTS5
    que romperian la sintaxis. Se extraen solo tokens alfanumericos, se quitan
    las palabras vacias y cada token se entrecomilla, por lo que FTS5 los trata
    como literales.
    """
    tokens = [
        t for t in _TOKEN.findall(texto.lower()) if len(t) >= 3 and t not in _VACIAS
    ]
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _insert_sync(docs: list[tuple[str, str, dict]]) -> int:
    filas = [(doc_id, content, json.dumps(meta, ensure_ascii=False)) for doc_id, content, meta in docs]
    db.executemany("INSERT OR IGNORE INTO docs(id, content, metadata) VALUES (?, ?, ?)", filas)
    return len(filas)


def _search_sync(query: str, k: int) -> list[dict]:
    match = _fts5_query(query)
    if match is None:
        return []
    rows = db.execute(
        """
        SELECT d.id, d.content, d.metadata, bm25(docs_fts) AS score
        FROM docs_fts
        JOIN docs d ON d.rowid = docs_fts.rowid
        WHERE docs_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (match, k),
    )
    return [
        {"id": r[0], "content": r[1], "metadata": json.loads(r[2]), "score": r[3]}
        for r in rows
    ]


def _get_by_ids_sync(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    marcadores = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT id, content, metadata FROM docs WHERE id IN ({marcadores})", tuple(ids)
    )
    return {r[0]: {"id": r[0], "content": r[1], "metadata": json.loads(r[2])} for r in rows}


def _existing_ids_sync(ids: list[str]) -> set[str]:
    if not ids:
        return set()
    marcadores = ",".join("?" * len(ids))
    rows = db.execute(f"SELECT id FROM docs WHERE id IN ({marcadores})", tuple(ids))
    return {r[0] for r in rows}


async def insert(docs: list[tuple[str, str, dict]]) -> int:
    """Inserta (id, content, metadata). Omite los ids que ya existen."""
    return await asyncio.to_thread(_insert_sync, docs)


async def search(query: str, k: int) -> list[dict]:
    """Busqueda lexical BM25. Devuelve id, content, metadata y score."""
    return await asyncio.to_thread(_search_sync, query, k)


async def get_by_ids(ids: list[str]) -> dict[str, dict]:
    """Recupera content y metadata por ID. Lo usa la busqueda semantica, que
    solo obtiene ids y distancias de sqlite-vec."""
    return await asyncio.to_thread(_get_by_ids_sync, ids)


async def existing_ids(ids: list[str]) -> set[str]:
    """Subconjunto de ids ya presentes: evita recalcular embeddings."""
    return await asyncio.to_thread(_existing_ids_sync, ids)


async def count() -> int:
    rows = await asyncio.to_thread(db.execute, "SELECT COUNT(*) FROM docs")
    return int(rows[0][0])


async def delete_all() -> None:
    await asyncio.to_thread(db.execute, "DELETE FROM docs")
