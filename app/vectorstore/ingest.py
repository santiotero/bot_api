"""Ingesta: texto crudo -> chunks -> FTS5 + sqlite-vec.

Todo el texto se normaliza a minusculas antes de cualquier operacion (chunking,
ID, indice lexical y embedding): evita duplicados por mayusculas, mejora el
matching lexical y reduce variacion en los embeddings.
"""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_config
from app.vectorstore import fts5_store, vec_store


def doc_id(content: str) -> str:
    """ID determinista: mismo texto -> mismo ID -> no se reinserta."""
    return hashlib.sha256(content[:200].encode("utf-8")).hexdigest()


def _splitter() -> RecursiveCharacterTextSplitter:
    cfg = get_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


async def ingest(documents: list[dict]) -> int:
    """Divide, deduplica e indexa. Devuelve la cantidad de chunks nuevos."""
    splitter = _splitter()

    # 1. Chunking sobre texto normalizado, deduplicando por ID dentro del lote.
    candidatos: dict[str, tuple[str, dict]] = {}
    for documento in documents:
        contenido = (documento.get("content") or "").lower().strip()
        if not contenido:
            continue
        metadata = documento.get("metadata") or {}
        for fragmento in splitter.split_text(contenido):
            fragmento = fragmento.strip()
            if fragmento:
                candidatos[doc_id(fragmento)] = (fragmento, metadata)

    if not candidatos:
        return 0

    # 2. Descartar los que ya estan indexados: no se recalcula su embedding.
    ya_existen = await fts5_store.existing_ids(list(candidatos))
    nuevos = {k: v for k, v in candidatos.items() if k not in ya_existen}
    if not nuevos:
        return 0

    ids = list(nuevos)
    contenidos = [nuevos[i][0] for i in ids]
    metadatas = [nuevos[i][1] for i in ids]

    # 3. Indexar en ambas stores con el mismo ID.
    await fts5_store.insert(list(zip(ids, contenidos, metadatas)))
    vectores = await vec_store.embed_passages(contenidos)
    await vec_store.upsert(list(zip(ids, vectores)))

    return len(ids)


async def clear() -> None:
    await fts5_store.delete_all()
    await vec_store.delete_all()


async def checkdb() -> dict:
    fts5_count = await fts5_store.count()
    vec_count = await vec_store.count()
    consistent = fts5_count == vec_count

    if fts5_count == 0 and vec_count == 0:
        status = "empty"
    elif not consistent:
        status = "inconsistent"
    else:
        status = "ok"

    return {
        "status": status,
        "fts5_count": fts5_count,
        "vec_count": vec_count,
        "consistent": consistent,
    }
