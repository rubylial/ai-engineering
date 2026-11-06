# The Intelligence System — atajos de operación.
# Los targets que usan `uv` corren en tu máquina (Python 3.12 + uv);
# los que terminan en `-docker` corren dentro del contenedor y solo necesitan Docker.

# Variables de .env disponibles como variables de make (no se exportan: los tests quedan herméticos).
-include .env

API_PORT ?= 8000
PHOENIX_PORT ?= 6006
REDIS_PORT ?= 6379
API_BASE_URL ?= http://localhost:$(API_PORT)
PHOENIX_UI_URL ?= http://localhost:$(PHOENIX_PORT)

.PHONY: help up down clean logs ps logs-trace ingest ingest-docker test test-integration lint format \
        scenarios scenarios-docker load graph eval evidence install

help:  ## Lista los comandos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependencias locales (uv) para tests, lint y scripts
	uv sync --frozen

up:  ## Levanta todo (valida .env, build, salud, ingesta)
	./run.sh

down:  ## Apaga los contenedores (conserva datos de Redis y Phoenix)
	docker compose down

clean:  ## Apaga y BORRA volúmenes (conversaciones y trazas)
	docker compose down -v

logs:  ## Logs JSON de la API en vivo
	docker compose logs -f api

ps:  ## Estado de los contenedores
	docker compose ps

logs-trace:  ## Líneas JSON de un request: make logs-trace REQ=req_... (correlación request_id/trace_id)
	@test -n "$(REQ)" || (echo "Uso: make logs-trace REQ=req_<id>"; exit 2)
	docker compose logs --no-log-prefix api | grep -F '$(REQ)' | python3 -m json.tool --no-ensure-ascii 2>/dev/null || \
	  docker compose logs --no-log-prefix api | grep -F '$(REQ)'

ingest:  ## Ingesta idempotente vía API
	curl -fsS -X POST "$(API_BASE_URL)/v1/ingest" -H "X-API-Key: $(API_KEY)" \
	  -H "Content-Type: application/json" -d '{"from_corpus": true}'; echo

ingest-docker:  ## Ingesta con el script, dentro del contenedor
	docker compose exec api python scripts/ingest.py

test:  ## Tests unitarios (sin red: LLM, Pinecone y Redis mockeados)
	uv run pytest -q

test-integration:  ## Tests contra el Redis del compose (y Pinecone si hay PINECONE_API_KEY)
	RUN_INTEGRATION=1 REDIS_URL=redis://localhost:$(REDIS_PORT)/0 uv run pytest -q -m integration

lint:  ## ruff check + formato
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:  ## Aplica formato
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

scenarios:  ## Corre los 6 escenarios y actualiza docs/traces/
	uv run python scripts/run_scenarios.py --base-url "$(API_BASE_URL)" --api-key "$(API_KEY)" --phoenix-url "$(PHOENIX_UI_URL)"

scenarios-docker:  ## Igual que scenarios, sin Python local
	docker compose run --rm --no-deps --user "$$(id -u):$$(id -g)" \
	  -v "$(CURDIR)/docs/traces:/app/docs/traces" \
	  -e PHOENIX_UI_URL="$(PHOENIX_UI_URL)" --entrypoint python api \
	  scripts/run_scenarios.py --base-url http://api:8000 --api-key "$(API_KEY)" --out-dir /app/docs/traces

load:  ## 5 peticiones concurrentes a /v1/chat: p50/p95/wall → docs/traces/load.md (centavos)
	uv run python scripts/load_smoke.py --n 5 --base-url "$(API_BASE_URL)" --api-key "$(API_KEY)" --phoenix-url "$(PHOENIX_UI_URL)"

graph:  ## Regenera el Mermaid del grafo real (docs/graph.mmd + README)
	uv run python scripts/export_graph.py

eval:  ## P@5 / R@5 / MRR global y por tipo: híbrido vs denso vs BM25 → docs/traces/evaluation.md
	uv run --env-file .env python scripts/evaluate_retrieval.py --k 5

evidence:  ## Regenera toda la evidencia con el stack arriba: scenarios + eval + load
	$(MAKE) scenarios
	$(MAKE) eval
	$(MAKE) load
	@echo "Evidencia en docs/traces/: traces.md, scenario_results.json, evaluation.md, load.md"

secrets:  ## Audita secretos (árbol + staged + commits sin pushear)
	python3 scripts/check_secrets.py
