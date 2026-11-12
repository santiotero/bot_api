"""RAG hibrido: BaseRetriever propio sobre la consulta SQL de hybrid_store.

La fusion lexical + semantica (RRF) la resuelve SQLite en una sola consulta
(ver app/vectorstore/hybrid_store.py), por lo que no hace falta un
EnsembleRetriever ni fusionar del lado de Python.
"""

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import get_config
from app.vectorstore import hybrid_store, vec_store


class HybridRetriever(BaseRetriever):
    """Recupera documentos combinando BM25 (FTS5) y coseno (sqlite-vec)."""

    k_semantico: int = 4
    k_lexical: int = 4
    score_threshold: float = 0.80
    pesos: list[float] = [0.5, 0.5]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise RuntimeError("Este retriever solo soporta invocacion asincronica")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        texto = query.lower().strip()
        if not texto:
            return []

        vector = await vec_store.embed_query(texto)
        filas = await hybrid_store.buscar(
            texto=texto,
            vector=vector,
            k_semantico=self.k_semantico,
            k_lexical=self.k_lexical,
            score_threshold=self.score_threshold,
            pesos=self.pesos,
        )
        return [
            Document(
                page_content=f["content"],
                metadata={**f["metadata"], "id": f["id"], "rrf": f["rrf"]},
            )
            for f in filas
        ]


def build_retriever() -> HybridRetriever:
    cfg = get_config()
    return HybridRetriever(
        k_semantico=cfg.k_semantico,
        k_lexical=cfg.k_lexical,
        score_threshold=cfg.score_threshold,
        pesos=cfg.ensemble_weights,
    )


async def busqueda_hibrida(consulta: str) -> str:
    """Busca informacion en la base de conocimiento del negocio.

    Args:
        consulta: texto a buscar.

    Returns:
        Los fragmentos encontrados, o 'sin resultados' si no hubo coincidencias.
    """
    documentos = await build_retriever().ainvoke(consulta)
    if not documentos:
        return "sin resultados"

    partes = []
    for i, doc in enumerate(documentos, start=1):
        fuente = doc.metadata.get("fuente") or doc.metadata.get("source") or "documento"
        partes.append(f"[{i}] ({fuente}) {doc.page_content}")
    return "\n\n".join(partes)
