"""Busqueda hibrida nativa en SQLite: sqlite-vec + FTS5 fusionados con RRF.

Una sola consulta resuelve todo: KNN vectorial, BM25 lexical y la fusion
Reciprocal Rank Fusion. Evita dos viajes a la base y la dependencia de un
retriever externo.

RRF: score(doc) = sum( peso_i / (K + rank_i) ) sobre cada ranking donde aparece.
Un documento que figura alto en ambos rankings supera a uno que domina solo uno.
"""

import asyncio
import json

from sqlite_vec import serialize_float32

from app.vectorstore import db
from app.vectorstore.fts5_store import _fts5_query

# Constante estandar de RRF: amortigua el peso de las primeras posiciones.
RRF_K = 60

_SQL_HIBRIDO = """
WITH vec_matches AS (
    SELECT doc_id, pos, similitud FROM (
        SELECT
            doc_id,
            ROW_NUMBER() OVER (ORDER BY distance) AS pos,
            1.0 - distance              AS similitud
        FROM vec_docs
        WHERE embedding MATCH ? AND k = ?
    )
    WHERE similitud >= ?
),
fts_matches AS (
    SELECT
        d.id AS doc_id,
        ROW_NUMBER() OVER (ORDER BY f.rank) AS pos
    FROM docs_fts f
    JOIN docs d ON d.rowid = f.rowid
    WHERE docs_fts MATCH ?
    LIMIT ?
),
candidatos AS (
    SELECT doc_id FROM vec_matches
    UNION
    SELECT doc_id FROM fts_matches
)
SELECT
    d.id,
    d.content,
    d.metadata,
    COALESCE(? / (? + f.pos), 0.0) + COALESCE(? / (? + v.pos), 0.0) AS rrf,
    v.similitud
FROM candidatos c
JOIN docs d            ON d.id = c.doc_id
LEFT JOIN vec_matches v ON v.doc_id = c.doc_id
LEFT JOIN fts_matches f ON f.doc_id = c.doc_id
ORDER BY rrf DESC
LIMIT ?
"""

_SQL_SOLO_VEC = """
WITH vec_matches AS (
    SELECT doc_id, pos, similitud FROM (
        SELECT
            doc_id,
            ROW_NUMBER() OVER (ORDER BY distance) AS pos,
            1.0 - distance              AS similitud
        FROM vec_docs
        WHERE embedding MATCH ? AND k = ?
    )
    WHERE similitud >= ?
)
SELECT
    d.id,
    d.content,
    d.metadata,
    ? / (? + v.pos) AS rrf,
    v.similitud
FROM vec_matches v
JOIN docs d ON d.id = v.doc_id
ORDER BY rrf DESC
LIMIT ?
"""


def _filas_a_docs(rows: list[tuple]) -> list[dict]:
    return [
        {
            "id": r[0],
            "content": r[1],
            "metadata": json.loads(r[2]),
            "rrf": r[3],
            "similitud": r[4],
        }
        for r in rows
    ]


def _buscar_sync(
    vector: list[float],
    texto: str,
    k_semantico: int,
    k_lexical: int,
    score_threshold: float,
    peso_lexical: float,
    peso_semantico: float,
    limite: int,
) -> list[dict]:
    embedding = serialize_float32(vector)
    match = _fts5_query(texto)

    # Sin tokens utiles para FTS5 (texto muy corto o solo simbolos) la parte
    # lexical no aporta y se consulta unicamente el indice vectorial.
    if match is None:
        rows = db.execute(
            _SQL_SOLO_VEC,
            (embedding, k_semantico, score_threshold, peso_semantico, RRF_K, limite),
        )
        return _filas_a_docs(rows)

    rows = db.execute(
        _SQL_HIBRIDO,
        (
            embedding, k_semantico, score_threshold,
            match, k_lexical,
            peso_lexical, RRF_K,
            peso_semantico, RRF_K,
            limite,
        ),
    )
    return _filas_a_docs(rows)


async def buscar(
    texto: str,
    vector: list[float],
    k_semantico: int,
    k_lexical: int,
    score_threshold: float,
    pesos: list[float],
    limite: int | None = None,
) -> list[dict]:
    """Busqueda hibrida con RRF.

    Args:
        texto: consulta ya normalizada a minusculas.
        vector: embedding de la consulta (prefijo 'query: ' ya aplicado).
        pesos: [peso_lexical, peso_semantico], en el orden de config.json.
    """
    peso_lexical, peso_semantico = pesos[0], pesos[1]
    if limite is None:
        limite = k_semantico + k_lexical

    return await asyncio.to_thread(
        _buscar_sync,
        vector, texto, k_semantico, k_lexical,
        score_threshold, peso_lexical, peso_semantico, limite,
    )
