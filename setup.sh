#!/usr/bin/env bash
#
# Instalacion completa. Idempotente: se puede volver a correr sin problema.
#
#   ./setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.12}

paso() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32mOK\033[0m %s\n' "$1"; }
aviso(){ printf '    \033[33m!\033[0m  %s\n' "$1"; }
error(){ printf '    \033[31mERROR\033[0m %s\n' "$1" >&2; }

# --------------------------------------------------------------------------- #
paso "1/5 Python"

if ! command -v "$PY" >/dev/null 2>&1; then
    error "no se encontro $PY"
    echo "    Ubuntu 24.04 lo trae nativo. En 22.04:"
    echo "      sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install -y python3.12 python3.12-venv"
    echo "    O indica otro interprete:  PYTHON=python3.11 ./setup.sh"
    exit 1
fi
ok "$($PY --version)"

# --------------------------------------------------------------------------- #
paso "2/5 Entorno virtual y dependencias"

# Se chequea el activate y no el directorio: un intento fallido deja un .venv
# incompleto, y darlo por bueno haria fallar el reintento de forma confusa.
if [ ! -f .venv/bin/activate ]; then
    [ -d .venv ] && { rm -rf .venv; aviso "se descarto un venv incompleto"; }
    "$PY" -m venv .venv || {
        rm -rf .venv
        error "fallo la creacion del venv"
        echo "    Falta el paquete del venv:  sudo apt install -y python3.12-venv"
        echo "    Despues volver a correr ./setup.sh"
        exit 1
    }
    ok "venv creado"
else
    ok "venv ya existe"
fi

# shellcheck disable=SC1091
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "dependencias instaladas"

# --------------------------------------------------------------------------- #
paso "3/5 Configuracion"

if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    chmod 600 config/.env
    aviso "se creo config/.env a partir del ejemplo"
    echo
    echo "    Hay que completarlo antes de seguir:  nano config/.env"
    echo "    Obligatorios: API_TOKEN, WATBOT_CALLBACK_TOKEN (mismo valor),"
    echo "                  API_PORT, REDIS_URL, LLAMACPP_BASE_URL"
    exit 1
fi

chmod 600 config/.env
python - <<'PY' || exit 1
import sys
try:
    from app.config import get_settings, get_config, get_prompts
    s = get_settings(); get_config(); get_prompts()
except Exception as exc:
    print(f"    ERROR en la configuracion: {exc}", file=sys.stderr)
    print("    Revisar config/.env", file=sys.stderr)
    sys.exit(1)

print(f"    OK API            {s.API_HOST}:{s.API_PORT}")
print(f"    OK LLM            {s.LLAMACPP_BASE_URL}")
print(f"    OK Redis          {s.REDIS_URL}")
if s.API_TOKEN != s.WATBOT_CALLBACK_TOKEN:
    print("    !  API_TOKEN y WATBOT_CALLBACK_TOKEN son distintos:")
    print("       watbot_v2 va a rechazar el callback")
else:
    print("    OK tokens         API_TOKEN == WATBOT_CALLBACK_TOKEN")
PY

# --------------------------------------------------------------------------- #
paso "4/5 Modelo de embeddings"

if python -c "
import sys
sys.path.insert(0,'.')
from app.vectorstore import vec_store
sys.exit(0 if vec_store._esta_en_cache() else 1)
" 2>/dev/null; then
    ok "ya esta en cache"
else
    echo "    Descargando (~470 MB, una sola vez)..."
    python scripts/download_model.py || {
        error "fallo la descarga o la verificacion del modelo"
        exit 1
    }
fi

# --------------------------------------------------------------------------- #
paso "5/5 Redis"

if python - <<'PY'
import asyncio, sys
sys.path.insert(0, '.')
from redis.asyncio import Redis
from app.config import get_settings

async def main():
    r = Redis.from_url(get_settings().REDIS_URL)
    try:
        await r.ping()
    finally:
        await r.aclose()

try:
    asyncio.run(main())
except Exception:
    sys.exit(1)
PY
then
    ok "responde"
else
    aviso "no responde: el worker no va a poder arrancar"
    echo "      sudo apt install -y redis-server"
    echo "    Y conviene activar AOF para que los datos sobrevivan un reinicio:"
    echo "      sudo sed -i 's/^appendonly no/appendonly yes/' /etc/redis/redis.conf"
    echo "      sudo systemctl restart redis-server"
fi

printf '\n\033[1mListo.\033[0m Para arrancar:  ./run.sh\n\n'
