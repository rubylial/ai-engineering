#!/usr/bin/env bash
# =============================================================================
# The Intelligence System — despliegue local con un solo comando.
#
#   ./run.sh               valida .env, construye, levanta, espera salud e ingesta el corpus
#   SKIP_INGEST=1 ./run.sh levanta sin ingestar
#
# Es idempotente: correrlo de nuevo reconstruye solo lo que cambió y la ingesta
# no vuelve a subir chunks que ya existen (dedup por hash de contenido).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

info() { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Herramientas ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || fail "Docker no está instalado."
docker compose version >/dev/null 2>&1 || fail "Se necesita Docker Compose v2 (comando 'docker compose')."
docker info >/dev/null 2>&1 || fail "El daemon de Docker no responde (¿Docker Desktop está abierto?)."
command -v curl >/dev/null 2>&1 || fail "Se necesita curl."

# --- 2. Configuración -------------------------------------------------------------
if [[ ! -f .env ]]; then
  cp .env.example .env
  fail "Creé .env a partir de .env.example. Completá ANTHROPIC_API_KEY, PINECONE_API_KEY y API_KEY y volvé a ejecutar ./run.sh"
fi

env_value() {  # lee una variable de .env sin ejecutar el archivo
  local line
  line="$(grep -E "^${1}=" .env | tail -n 1 || true)"
  line="${line#*=}"
  line="${line%$'\r'}"   # .env editado en Windows (CRLF)
  line="${line%\"}"; line="${line#\"}"
  printf '%s' "$line"
}

LLM_PROVIDER="$(env_value LLM_PROVIDER)"; LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
VECTOR_STORE="$(env_value VECTOR_STORE)"; VECTOR_STORE="${VECTOR_STORE:-pinecone}"
API_PORT="$(env_value API_PORT)"; API_PORT="${API_PORT:-8000}"
PHOENIX_PORT="$(env_value PHOENIX_PORT)"; PHOENIX_PORT="${PHOENIX_PORT:-6006}"
API_KEY="$(env_value API_KEY)"

required=(API_KEY)
[[ "$LLM_PROVIDER" == "anthropic" ]] && required+=(ANTHROPIC_API_KEY)
[[ "$LLM_PROVIDER" == "openai" ]] && required+=(OPENAI_API_KEY)
[[ "$VECTOR_STORE" == "pinecone" ]] && required+=(PINECONE_API_KEY)
missing=()
for key in "${required[@]}"; do
  [[ -n "$(env_value "$key")" ]] || missing+=("$key")
done
(( ${#missing[@]} == 0 )) || fail "Faltan valores en .env: ${missing[*]}"
[[ "$API_KEY" == "cambia-esta-clave-larga-y-secreta" ]] && warn "API_KEY sigue con el valor de ejemplo: cambiala si vas a exponer la API."
ok "Configuración válida (LLM=${LLM_PROVIDER}, vector store=${VECTOR_STORE})"

# --- 3. Levantar y esperar salud -------------------------------------------------
info "Construyendo la imagen y levantando redis, phoenix y api (la primera vez tarda unos minutos)..."
if ! docker compose up -d --build --wait --wait-timeout 300; then
  docker compose ps
  docker compose logs --tail=60 api
  fail "Algún servicio no quedó saludable. Revisá los logs de arriba."
fi
ok "Contenedores saludables"

# --- 4. Readiness: Redis, Pinecone y LLM -----------------------------------------
info "Verificando dependencias externas (GET /ready)..."
ready_body="$(mktemp)"
trap 'rm -f "$ready_body"' EXIT
status="$(curl -s -o "$ready_body" -w '%{http_code}' "http://localhost:${API_PORT}/ready" || true)"
if [[ "$status" != "200" ]]; then
  cat "$ready_body"; echo
  fail "/ready respondió ${status}. El detalle de arriba indica qué componente falla (clave inválida, índice incompatible, etc.)."
fi
ok "Redis, Pinecone y el LLM responden"

# --- 5. Ingesta idempotente ---------------------------------------------------------
if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
  info "Ingesta idempotente del corpus (la segunda vez sube 0 chunks)..."
  curl -fsS -X POST "http://localhost:${API_PORT}/v1/ingest" \
    -H "X-API-Key: ${API_KEY}" -H "Content-Type: application/json" \
    -d '{"from_corpus": true}' || fail "La ingesta falló: revisá 'docker compose logs api'."
  echo
  ok "Corpus indexado"
fi

cat <<EOF

$(ok "The Intelligence System está arriba")
  API (Swagger):  http://localhost:${API_PORT}/docs
  Phoenix:        http://localhost:${PHOENIX_PORT}
  Probar:         curl -s -X POST http://localhost:${API_PORT}/v1/chat \\
                    -H "X-API-Key: \$API_KEY" -H "Content-Type: application/json" \\
                    -d '{"message": "¿Qué es BM25?"}'
  Escenarios:     make scenarios
  Apagar:         make down
EOF
