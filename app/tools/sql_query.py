"""Text-to-SQL acotado a la tabla de catalogo, con defensa en capas.

Capas de proteccion contra SQL injection:
  1. El texto del usuario llega al LLM envuelto en <consulta_usuario>, nunca se
     concatena directo en la sentencia.
  2. La sentencia generada se valida aca: un solo statement, solo SELECT, solo
     TABLA_PERMITIDA, sin comentarios ni palabras prohibidas.
  3. El usuario MySQL (DB_BUSCADOR_USER) tiene unicamente SELECT sobre esta
     tabla. Es el limite duro: si las capas 1 y 2 fallaran, la base rechaza.
  4. Timeout de ejecucion y LIMIT forzado.

TABLA_PERMITIDA es una constante y no un parametro de configuracion a
proposito: es un limite de seguridad y no debe poder ampliarse por config.
"""

import asyncio
import re
import time

import sqlparse
from sqlparse.sql import Statement

TABLA_PERMITIDA = "catalog_items"
LIMITE_FILAS = 10
TIMEOUT_SEGUNDOS = 5

# El sondeo de disponibilidad es mas corto que una consulta real: solo pregunta
# si hay filas. Si la base no es alcanzable se paga este timeout completo.
TIMEOUT_SONDEO = 3
# ...y se paga una vez por minuto, no una vez por mensaje: cachear acota tanto
# el costo cuando la base no responde como los viajes a MySQL cuando si responde.
# El precio es que un catalogo recien cargado tarda hasta un minuto en verse.
CACHE_SONDEO_SEGUNDOS = 60

# Solo palabras que podrian pasar las otras capas. Quedan fuera a proposito
# terminos como user, database, version, set o call: no son SELECT o no
# referencian tabla, por lo que ya los rechazan las validaciones
# estructurales, y son palabras plausibles en la descripcion de un producto
# (bloquearlas rompería busquedas legitimas con LIKE).
PALABRAS_PROHIBIDAS = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "create", "truncate",
        "replace", "grant", "revoke", "exec", "execute",
        "union", "intersect", "except", "into", "load_file", "outfile",
        "dumpfile", "sleep", "benchmark", "information_schema", "mysql",
        "performance_schema",
    }
)

_TABLAS = re.compile(r"\b(?:from|join)\s+[`\"']?(\w+)[`\"']?", re.IGNORECASE)
_PALABRA = re.compile(r"\b\w+\b")


class SQLNoPermitido(ValueError):
    pass


def _quitar_markdown(sql: str) -> str:
    """El LLM suele devolver la sentencia dentro de un bloque de codigo."""
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";").strip()


def validar(sql: str) -> str:
    """Valida la sentencia y devuelve la version segura a ejecutar."""
    sql = _quitar_markdown(sql)
    if not sql:
        raise SQLNoPermitido("sentencia vacia")

    if "--" in sql or "/*" in sql or "#" in sql:
        raise SQLNoPermitido("no se permiten comentarios")
    if ";" in sql:
        raise SQLNoPermitido("no se permiten sentencias multiples")

    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        raise SQLNoPermitido("se espera una sola sentencia")

    statement: Statement = statements[0]
    if statement.get_type() != "SELECT":
        raise SQLNoPermitido("solo se permiten consultas SELECT")

    minuscula = sql.lower()
    encontradas = set(_PALABRA.findall(minuscula)) & PALABRAS_PROHIBIDAS
    if encontradas:
        raise SQLNoPermitido(f"palabra no permitida: {sorted(encontradas)[0]}")

    tablas = {t.lower() for t in _TABLAS.findall(sql)}
    if not tablas:
        raise SQLNoPermitido("la consulta no referencia ninguna tabla")
    if tablas != {TABLA_PERMITIDA}:
        prohibida = sorted(tablas - {TABLA_PERMITIDA})[0]
        raise SQLNoPermitido(f"solo se permite consultar {TABLA_PERMITIDA}, no {prohibida}")

    if not re.search(r"\blimit\s+\d+", minuscula):
        sql = f"{sql} LIMIT {LIMITE_FILAS}"

    return sql


async def _ejecutar(sql: str) -> list[dict]:
    import aiomysql

    from app.config import get_settings

    settings = get_settings()
    conexion = await aiomysql.connect(
        host=settings.DB_HOST,
        db=settings.DB_NAME,
        user=settings.DB_BUSCADOR_USER,
        password=settings.DB_BUSCADOR_PASS,
        charset="utf8mb4",
        connect_timeout=TIMEOUT_SEGUNDOS,
        autocommit=True,
    )
    try:
        async with conexion.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(sql)
            return await cursor.fetchall()
    finally:
        conexion.close()


_sondeo_cache: tuple[float, bool] | None = None


async def catalogo_tiene_filas() -> bool:
    """Si la tabla esta vacia o no es alcanzable, el supervisor no rutea al buscador.

    Se consulta al inicio de cada mensaje, pero el resultado se cachea: el estado
    del catalogo cambia con poca frecuencia y una base inalcanzable costaria el
    timeout completo en cada mensaje.
    """
    global _sondeo_cache

    ahora = time.monotonic()
    if _sondeo_cache is not None and ahora - _sondeo_cache[0] < CACHE_SONDEO_SEGUNDOS:
        return _sondeo_cache[1]

    try:
        filas = await asyncio.wait_for(
            _ejecutar(f"SELECT 1 FROM {TABLA_PERMITIDA} WHERE activo = 1 LIMIT 1"),
            timeout=TIMEOUT_SONDEO,
        )
        disponible = bool(filas)
    except Exception:
        # Sin acceso a la base el buscador no puede aportar: se trata como vacio.
        disponible = False

    _sondeo_cache = (ahora, disponible)
    return disponible


async def sql_query(consulta_sql: str) -> str:
    """Ejecuta una consulta SELECT sobre la tabla catalog_items.

    Args:
        consulta_sql: sentencia SELECT. Solo se permite la tabla catalog_items.

    Returns:
        Las filas encontradas, 'sin resultados', o un mensaje que empieza con
        ERROR si la sentencia fue rechazada o fallo.
    """
    try:
        segura = validar(consulta_sql)
    except SQLNoPermitido as exc:
        return f"ERROR: {exc}"

    try:
        filas = await asyncio.wait_for(_ejecutar(segura), timeout=TIMEOUT_SEGUNDOS)
    except asyncio.TimeoutError:
        return "ERROR: la consulta excedio el tiempo maximo"
    except Exception as exc:
        return f"ERROR: fallo la consulta ({type(exc).__name__})"

    if not filas:
        return "sin resultados"

    lineas = []
    for fila in filas:
        campos = [f"{k}={v}" for k, v in fila.items() if v is not None]
        lineas.append(" | ".join(campos))
    return "\n".join(lineas)
