# api-agente — Pre-entrega 7

API FastAPI asíncrona que expone el orquestador multi-agente del Módulo 6 como un
servicio de producción: encola tareas y devuelve un `job_id` sin bloquear, persiste
el estado en Redis, traza cada ejecución en Arize Phoenix y pausa las tareas
críticas para aprobación humana (HITL) con `interrupt()` de LangGraph.

## Arquitectura

```
                 POST /tasks                      ┌──────────────┐
   cliente ───────────────────────►  FastAPI  ──► │ asyncio.Queue│
      ▲         (devuelve job_id)       │         └──────┬───────┘
      │  GET /tasks/{id}                │                │  (no bloquea el event loop)
      └─────────────────────────────── │ ◄──┐      ┌─────▼──────┐
                 POST /tasks/{id}/approve     │      │  Worker    │ (pool de N corrutinas)
                                              │      │  pool      │
                                        ┌─────┴──┐   └─────┬──────┘
                                        │ Redis  │◄────────┤ corre el grafo del M6
                                        │ jobs + │   ┌─────▼──────────────────────┐
                                        │ check- │   │ gate(HITL) → supervisor →  │
                                        │ points │   │ researcher / analyst       │
                                        └────────┘   └─────┬──────────────────────┘
                                                           │ trazas OTLP
                                                     ┌─────▼──────┐
                                                     │  Phoenix   │
                                                     └────────────┘
```

- **La API nunca corre el grafo en el request.** `POST /tasks` crea el job en Redis
  (estado `PENDING`), lo encola y devuelve el `job_id`. Un pool de workers consume la
  cola y ejecuta el grafo en segundo plano, actualizando el estado en Redis
  (`RUNNING → DONE | FAILED | WAITING_APPROVAL`).
- **Cualquier excepción del agente deja el job en `FAILED`** (con el mensaje de error),
  así el cliente que hace polling nunca se queda esperando para siempre.
- **El estado vive en Redis dos veces:** el estado del *job* (JobStore) y los
  *checkpoints* de LangGraph (`AsyncRedisSaver`). Los checkpoints son lo que hace
  posible pausar en el HITL y reanudar más tarde, incluso tras reiniciar la API.

## Estructura

```
app/
  main.py            # FastAPI + lifespan (arma Redis, checkpointer, grafo, worker, Phoenix)
  routes.py          # endpoints: POST /tasks, GET /tasks/{id}, POST /tasks/{id}/approve
  graph.py           # grafo del M6 + nodo gate (HITL) + checkpointer inyectable
  hitl.py            # is_critical() + gate_node con interrupt()
  worker.py          # pool de workers: consume la cola, corre el grafo, actualiza Redis
  jobs.py            # JobStore en Redis (PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED)
  observability.py   # init de Arize Phoenix + OpenInference (traza automática)
  state.py           # SupervisorState (del M6 + campos HITL) + SupervisorDecision
  supervisor.py      # supervisor del M6 (step budget + anti-repetición)
  agents/            # researcher (KB local) + analyst (tools de cómputo) — del M6
scripts/
  load_test.py       # 5 peticiones concurrentes + latencias
tests/               # 22 tests sin red (fakeredis + InMemorySaver): HITL, worker, API, jobs
requirements.txt     # dependencias congeladas (también hay pyproject.toml + uv.lock)
docker-compose.yml   # api + redis 8 + phoenix
Dockerfile
.env.example
screenshots/         # acá van las capturas de Phoenix (evidencia)
```

## Cómo levantarlo

### Opción A — Docker Compose (recomendada, todo junto)

```bash
cp .env.example .env            # completá ANTHROPIC_API_KEY
docker compose up -d --build
curl localhost:8000/ready       # {"status":"ready"}
```

Levanta `redis` (8.10.3, con AOF), `phoenix` (UI en http://localhost:6006) y `api`
(http://localhost:8000). La API espera a que Redis y Phoenix estén healthy.

### Opción B — Local (API en tu máquina, Redis y Phoenix en Docker)

```bash
docker run -d --name redis -p 6379:6379 redis:8.10.3 redis-server --appendonly yes
docker run -d --name phoenix -p 6006:6006 arizephoenix/phoenix:version-21.0.0

cp .env.example .env            # completá ANTHROPIC_API_KEY
uv sync                         # o: pip install -r requirements.txt
cp .env.example .env
uv run uvicorn app.main:create_app --factory --reload
```

> **Redis tiene que ser 8+**: `AsyncRedisSaver` necesita los módulos RedisJSON y
> RediSearch, que la imagen `redis:8.x` trae de fábrica. Un Redis 7 o `redis-stack`
> viejo falla con errores `FT.` / `JSON.`.

## Probar la API

```bash
BASE=http://localhost:8000

# 1. Tarea normal → se procesa sola
JOB=$(curl -s -X POST $BASE/tasks -H 'Content-Type: application/json' \
  -d '{"query":"¿Qué dice la doc sobre el patrón supervisor en LangGraph?"}' | jq -r .job_id)
curl -s $BASE/tasks/$JOB | jq        # status: PENDING → RUNNING → DONE

# 2. Tarea CRÍTICA → se pausa esperando aprobación (HITL)
JOB=$(curl -s -X POST $BASE/tasks -H 'Content-Type: application/json' \
  -d '{"query":"ejecutar el borrado de la tabla en producción"}' | jq -r .job_id)
curl -s $BASE/tasks/$JOB | jq        # status: WAITING_APPROVAL, con el campo "interrupt"

# Aprobar (o rechazar con approved:false)
curl -s -X POST $BASE/tasks/$JOB/approve -H 'Content-Type: application/json' \
  -d '{"approved":true}' | jq
curl -s $BASE/tasks/$JOB | jq        # status: DONE

# También podés forzar el HITL en cualquier query:
curl -s -X POST $BASE/tasks -H 'Content-Type: application/json' \
  -d '{"query":"consulta inocente","require_approval":true}'
```

Una tarea es **crítica** si su texto contiene alguna palabra de `CRITICAL_KEYWORDS`
(configurable en `.env`: ejecutar, borrar, producción, pago, deploy…) o si el request
manda `require_approval: true`.

## Prueba de carga: 5 peticiones concurrentes

```bash
# Subí la concurrencia del worker para que las 5 corran en paralelo:
#   en .env → WORKER_CONCURRENCY=5, y recreá la API (docker compose up -d api)
uv run python scripts/load_test.py --base-url http://localhost:8000 --n 5
```

El script lanza 5 `POST /tasks` con `asyncio.gather`, hace polling de cada uno hasta
`DONE` e imprime las latencias (incluido un p95 local como sanity check). Las queries
son no-críticas a propósito, para que completen sin pausa.

### Qué capturar en `screenshots/` (evidencia de la rúbrica)

Abrí Phoenix (http://localhost:6006), proyecto **api-agente**, y después de correr el
load test guardá:

| Archivo | Qué tiene que verse |
| --- | --- |
| `01-trazas.png` | La lista de trazas con las 5 ejecuciones de la corrida concurrente |
| `02-arbol-traza.png` | Una traza abierta: árbol `gate → supervisor → researcher/analyst`, con los spans de cada llamada al LLM |
| `03-costo-por-ejecucion.png` | La columna/panel de **costo por ejecución** (Phoenix lo calcula de los tokens) |
| `04-latencia-p95.png` | La métrica de **latencia p95** de la corrida |
| `05-hitl.png` | (opcional) La traza de una tarea crítica, mostrando la pausa del HITL |

> **Sobre el costo en Phoenix:** Phoenix calcula el costo solo si tiene el precio del
> modelo en su tabla. `claude-haiku-5-5` es nuevo y puede no estar todavía: si la
> columna de costo sale vacía, tenés tres salidas — (a) correr el load test con un
> modelo que Phoenix ya conozca (`LLM_MODEL=claude-haiku-4-5`), (b) cargar el precio a
> mano en Phoenix (Settings → Models), o (c) sacar los tokens de entrada/salida de la
> traza y multiplicar por el precio de la tabla de Anthropic. Probabilidad de que el
> costo aparezca solo con haiku-5-5: ~40%; con haiku-4-5: ~70%.

## Observabilidad

`app/observability.py` registra el TracerProvider de Phoenix e instrumenta LangChain
con OpenInference. **No hace falta decorar funciones a mano**: el `LangChainInstrumentor`
traza automáticamente cada nodo del grafo y cada llamada al LLM. El worker envuelve
cada job en `session.id = job_id`, así en la vista **Sessions** de Phoenix todas las
trazas de un mismo job (incluido el turno de aprobación) quedan agrupadas.

## Checklist de la entrega → dónde está

| Ítem | Dónde |
| --- | --- |
| Endpoint async que encola y devuelve job_id sin bloquear | `routes.py::create_task` + `worker.py` (cola + pool) |
| Estado en Redis, incluido el paso a FAILED ante excepción | `jobs.py::JobStore` + `worker.py::_run` (try/except → FAILED) |
| Trazas en el dashboard | `observability.py` (Phoenix + OpenInference) |
| Costo por ejecución y p95 de 5 concurrentes | `scripts/load_test.py` + capturas en `screenshots/` |
| Nodo Human-in-the-loop que pausa hasta aprobación | `hitl.py::gate_node` (`interrupt()`) + `POST /tasks/{id}/approve` |
| README + requirements.txt | este archivo + `requirements.txt` |
| Repo clonable en limpio | `git clone`, `uv sync` (o `pip install -r requirements.txt`), `pytest -q` |

## Tests

```bash
uv run pytest -q    # 22 passed, sin red (fakeredis + InMemorySaver)
uv run ruff check . # lint limpio
```

Cubre: detección de criticidad y el gate (pausa/aprobación/rechazo con `interrupt`);
el JobStore sobre fakeredis; el worker end-to-end (DONE, WAITING_APPROVAL→approve,
rechazo, y **FAILED ante una excepción del agente**); y los endpoints de la API
(encolado no bloqueante, 404, flujo HITL completo, 409 al aprobar un job que no está
esperando). El grafo de los tests usa el `gate_node` real pero reemplaza el LLM por
nodos fake, así corre sin claves ni Redis real.

## Notas y límites

- **Worker pattern**: pool de corrutinas sobre una `asyncio.Queue` (lo que el enunciado
  llama "thread asíncrono simple"). Para producción real se cambiaría por Celery/Arq
  sin tocar los endpoints: solo el `Worker`.
- **Costo del demo**: cada tarea son 3–7 llamadas a Haiku; el load test de 5, unas
  15–35 llamadas. Centavos.
- El orquestador (supervisor + researcher + analyst, KB local) es el del Módulo 6; lo
  único nuevo acá es la capa de producción (API, Redis, Phoenix, HITL).
