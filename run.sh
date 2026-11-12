#!/usr/bin/env bash
#
# Levanta la API y el worker juntos, en primer plano.
# Ctrl+C corta los dos.
#
#   ./run.sh
#
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Falta el entorno virtual. Correr primero:  ./setup.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate

# Se leen con grep y no con 'source': los valores del .env pueden tener espacios
# y caracteres que la shell interpretaria.
leer_env() { grep -E "^$1=" config/.env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' ; }
HOST=$(leer_env API_HOST); HOST=${HOST:-0.0.0.0}
PORT=$(leer_env API_PORT); PORT=${PORT:-8000}

# Al salir se mata todo el grupo de procesos, para no dejar el worker huerfano.
trap 'kill 0' EXIT INT TERM

echo "API    -> http://$HOST:$PORT"
echo "worker -> arq"
echo

uvicorn app.main:app --host "$HOST" --port "$PORT" &
arq app.worker.WorkerSettings &

wait -n
echo
echo "Uno de los procesos termino: se cortan los dos." >&2
