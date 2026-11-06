# entity-pipeline — Pre-entrega 2

Pipeline LCEL que recibe un párrafo de texto y devuelve un objeto Pydantic validado con las
tecnologías mencionadas, el nivel de criticidad y un resumen técnico. Reintenta
automáticamente si el LLM devuelve un JSON mal formado o incompleto.

## Qué hay adentro

```
src/entity_pipeline/
  schemas.py   # contrato Pydantic v2: TechExtraction + NivelCriticidad
  chain.py     # LCEL: prompt | model.with_structured_output(Schema) + .with_retry()
scripts/
  demo.py      # mini script asíncrono con 3 textos de ejemplo
tests/         # 15+ tests unitarios sin red (schema, cadena, reintentos)
```

## Cómo correrlo

```bash
uv sync                               # instala langchain-anthropic, pydantic, pytest…
cp .env.example .env                  # completá ANTHROPIC_API_KEY
uv run python scripts/demo.py         # 3 textos de ejemplo (~3 llamadas a Haiku)
uv run pytest -q                      # tests sin red
uv run ruff check .                   # lint limpio
```

Cambiar el modelo: `LLM_MODEL=claude-sonnet-5-5 uv run python scripts/demo.py`.

## Ejemplo de entrada y salida

Entrada:

```text
API en FastAPI con caché en Redis y persistencia en PostgreSQL. El endpoint /checkout
está cayendo bajo carga: p95 de 4s y 503 recurrentes. La cola de conexiones del pool
a Postgres se satura y los workers de Uvicorn se quedan esperando.
```

Salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "Uvicorn"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Cómo se arma la cadena (`chain.py`)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from entity_pipeline.schemas import TechExtraction

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Sos un asistente técnico. Extraés tecnologías, criticidad y resumen…"),
        ("human", "{text}"),
    ]
)

model = ChatAnthropic(model="claude-haiku-5-5", temperature=0, max_retries=0)

chain = (prompt | model.with_structured_output(TechExtraction)).with_retry(
    retry_if_exception_type=(ValidationError, OutputParserException),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

result: TechExtraction = await chain.ainvoke({"text": "..."})
```

- **Sin f-strings en el prompt:** `ChatPromptTemplate` administra la variable `{text}`
  y queda modular para agregar more inputs (ej. instrucciones de formato variables).
- **`with_structured_output()`** fuerza al modelo a devolver el schema vía tool
  calling (recomendado para Anthropic y OpenAI).
- **`.with_retry()`** reintenta con backoff exponencial solo ante `ValidationError`
  o `OutputParserException`. Un 401 o un 429 no se reintenta.

## Decisiones

- **Modelo por defecto `claude-haiku-5-5`:** activo en la página de deprecaciones de
  Anthropic al 2026-11-05. Se cambia por variable de entorno `LLM_MODEL`.
- **`max_retries=0` en `ChatAnthropic`:** el SDK tiene retries propios; los apagamos
  para no duplicar los de `.with_retry()`.
- **Detección de respuestas incompletas:** cuando el LLM corta por `max_tokens`, la
  tool call llega truncada; `with_structured_output()` levanta `OutputParserException`
  y la cadena reintenta. Los validators del schema (lista no vacía, enum, largo del
  resumen) atrapan lo que el parser deja pasar.
- **Logs:** `logging` estándar de la stdlib, formato clave=valor. Cada request
  loguea `pipeline.start`, un posible `pipeline.validation_error` o
  `pipeline.parser_error` (antes de que `.with_retry()` reintente), y `pipeline.ok`
  al final.

## Tests

`uv run pytest -q` corre todos los tests sin red. El LLM se reemplaza por un
`Runnable` fake que devuelve respuestas pre-armadas; así se verifica que:

- el schema rechaza listas vacías, criticidad inválida, resumen muy largo y colapsa
  espacios;
- la cadena reintenta cuando el fake levanta `OutputParserException`;
- no reintenta cuando la excepción es de otra familia (ej. un `RuntimeError` por 401);
- `process_text` rechaza entradas vacías con `ValueError`.
