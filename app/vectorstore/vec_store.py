"""Busqueda semantica (sqlite-vec) y generacion de embeddings.

El modelo e5 requiere prefijos: "passage: " al indexar y "query: " al buscar.
Omitirlos degrada la calidad de la busqueda de forma notoria.
"""

import asyncio
import threading

from sqlite_vec import serialize_float32

from app.vectorstore import db

_model = None
_model_lock = threading.Lock()
_registrado = False


def _registrar_modelo() -> str:
    """Registra e5-small en fastembed y devuelve su nombre.

    No esta en el catalogo fijo de fastembed (que solo trae e5-large, de 1024
    dimensiones), pero el repo oficial de HuggingFace ya publica los ONNX
    preconvertidos, asi que no hay que exportar nada.

    pooling=MEAN y normalization=True salen de 1_Pooling/config.json del propio
    modelo. Un pooling distinto no falla: produce vectores validos pero
    semanticamente incorrectos, y el error pasaria desapercibido.
    """
    global _registrado
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    from app.config import get_config

    nombre = get_config().embedding_model
    if not _registrado:
        TextEmbedding.add_custom_model(
            model=nombre,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=nombre),
            dim=db.EMBEDDING_DIM,
            model_file="onnx/model.onnx",
        )
        _registrado = True
    return nombre


def _construir_modelo(local_files_only: bool):
    from fastembed import TextEmbedding

    from app.config import get_settings

    nombre = _registrar_modelo()
    return TextEmbedding(
        model_name=nombre,
        cache_dir=get_settings().EMBEDDING_CACHE_DIR,
        local_files_only=local_files_only,
    )


def _get_model():
    """Carga perezosa del modelo (singleton por proceso), siempre offline.

    local_files_only=True evita que fastembed consulte HuggingFace en cada
    arranque para validar el cache: sin eso, que HF no responda impide arrancar
    el worker. El modelo se baja una sola vez con scripts/download_model.py.
    """
    global _model
    with _model_lock:
        if _model is None:
            from app.config import get_settings

            try:
                _model = _construir_modelo(local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "El modelo de embeddings no esta en cache "
                    f"({get_settings().EMBEDDING_CACHE_DIR}). "
                    "Ejecutar una vez: python scripts/download_model.py"
                ) from exc
        return _model


def descargar_modelo() -> None:
    """Baja el modelo al cache. Unico punto que accede a la red."""
    _construir_modelo(local_files_only=False)


def _esta_en_cache() -> bool:
    """True si el modelo ya se puede cargar sin red. Lo usa setup.sh."""
    try:
        _construir_modelo(local_files_only=True)
        return True
    except Exception:
        return False


def _encode_sync(textos: list[str], prefijo: str) -> list[list[float]]:
    modelo = _get_model()
    entradas = [prefijo + t for t in textos]
    # normalization=True en el registro: los vectores ya vienen unitarios.
    return [v.tolist() for v in modelo.embed(entradas)]


def _upsert_sync(items: list[tuple[str, list[float]]]) -> int:
    filas = [(doc_id, serialize_float32(vec)) for doc_id, vec in items]
    # vec0 no soporta INSERT OR IGNORE: ante un doc_id repetido aborta con
    # UNIQUE constraint. Se borra primero para que el upsert sea idempotente.
    # El trabajo caro (embeddings) ya se evita en ingest.py, que descarta los
    # ids existentes antes de llegar hasta aca.
    db.executemany("DELETE FROM vec_docs WHERE doc_id = ?", [(doc_id,) for doc_id, _ in items])
    db.executemany("INSERT INTO vec_docs(doc_id, embedding) VALUES (?, ?)", filas)
    return len(filas)


def _search_sync(vector: list[float], k: int, score_threshold: float) -> list[dict]:
    rows = db.execute(
        """
        SELECT doc_id, distance
        FROM vec_docs
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (serialize_float32(vector), k),
    )
    # distance es coseno: 0 identico, 2 opuesto. similitud = 1 - distance.
    resultados = []
    for doc_id, distance in rows:
        similitud = 1.0 - float(distance)
        if similitud >= score_threshold:
            resultados.append({"id": doc_id, "similitud": similitud})
    return resultados


async def embed_passages(textos: list[str]) -> list[list[float]]:
    """Embeddings para indexar (prefijo 'passage: ')."""
    return await asyncio.to_thread(_encode_sync, textos, "passage: ")


async def embed_query(texto: str) -> list[float]:
    """Embedding para buscar (prefijo 'query: ')."""
    vectores = await asyncio.to_thread(_encode_sync, [texto], "query: ")
    return vectores[0]


async def upsert(items: list[tuple[str, list[float]]]) -> int:
    """Inserta (id, embedding). Omite los ids que ya existen."""
    return await asyncio.to_thread(_upsert_sync, items)


async def search(query: str, k: int, score_threshold: float) -> list[dict]:
    """Busqueda semantica. Devuelve ids que superan el umbral de similitud."""
    vector = await embed_query(query)
    return await asyncio.to_thread(_search_sync, vector, k, score_threshold)


async def count() -> int:
    rows = await asyncio.to_thread(db.execute, "SELECT COUNT(*) FROM vec_docs")
    return int(rows[0][0])


async def delete_all() -> None:
    await asyncio.to_thread(db.execute, "DELETE FROM vec_docs")


async def warmup() -> None:
    """Fuerza la carga del modelo para que el primer request no pague el costo."""
    await asyncio.to_thread(_get_model)
