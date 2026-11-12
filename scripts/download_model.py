"""Descarga el modelo de embeddings al cache local. Se corre una sola vez.

Es el unico punto del sistema que accede a HuggingFace. En produccion el modelo
se carga con local_files_only=True, para que un HuggingFace caido o lento no
impida arrancar la API ni el worker.

    python scripts/download_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_config, get_settings  # noqa: E402
from app.vectorstore import db, vec_store  # noqa: E402


def main() -> int:
    nombre = get_config().embedding_model
    destino = get_settings().EMBEDDING_CACHE_DIR
    print(f"Descargando {nombre} en {destino} ...")

    vec_store.descargar_modelo()

    # Verificacion: un modelo mal configurado (pooling o dimension equivocados)
    # no falla al cargar, devuelve vectores invalidos en silencio.
    vector = vec_store._encode_sync(["prueba de verificacion"], "query: ")[0]
    norma = sum(x * x for x in vector) ** 0.5

    print(f"  dimensiones: {len(vector)} (esperado {db.EMBEDDING_DIM})")
    print(f"  norma L2:    {norma:.6f} (esperado 1.0, los vectores deben venir normalizados)")

    if len(vector) != db.EMBEDDING_DIM:
        print(f"ERROR: el modelo devuelve {len(vector)} dimensiones y el esquema espera {db.EMBEDDING_DIM}")
        return 1
    if abs(norma - 1.0) > 0.01:
        print("ERROR: los vectores no estan normalizados; revisar normalization=True en el registro")
        return 1

    print("OK: modelo en cache y verificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
