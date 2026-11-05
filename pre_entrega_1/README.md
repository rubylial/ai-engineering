# Pre-entrega 1 — Unified Async LLM Client

Cliente asíncrono unificado para OpenAI y Anthropic bajo una única interfaz. Soporta modo normal y **streaming de tokens**, con validación Pydantic, manejo de secretos vía `SecretStr` y captura controlada de errores de red/API.

## Objetivos cubiertos

| Requisito | Dónde se resuelve |
|---|---|
| Intercambiabilidad entre proveedores | `llm_client/base.py` + `llm_client/manager.py` |
| Asincronía (no bloqueante) | `AsyncOpenAI` y `AsyncAnthropic` en los providers |
| Streaming (generador asíncrono) | método `stream()` con `yield` en cada provider |
| Validación Pydantic (entrada + config) | `llm_client/schemas.py` |
| Manejo de secretos seguro | `SecretStr` en `ManagerConfig` |
| Config vía variables de entorno | `ManagerConfig.from_env()` + `python-dotenv` |
| Captura de errores (auth, rate limit, red) | `llm_client/exceptions.py` + try/except en `main.py` |

## Estructura

```
pre-entrega-1/
├── llm_client/
│   ├── __init__.py
│   ├── base.py                  # ABC: contrato común
│   ├── manager.py               # Fachada AsyncLLMManager
│   ├── schemas.py               # Pydantic: ChatMessage, ModelConfig, ModelResponse
│   ├── exceptions.py            # Jerarquía de errores
│   └── providers/
│       ├── __init__.py
│       ├── openai_client.py     # Implementación OpenAI
│       └── anthropic_client.py  # Implementación Anthropic
├── main.py                      # Script de validación (normal + streaming)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Requiere **Python 3.12+**.

```bash
# 1. Crear entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# editar .env con tus keys reales
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | sí | `openai` o `anthropic` |
| `OPENAI_API_KEY` | si `LLM_PROVIDER=openai` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | si `LLM_PROVIDER=anthropic` | API key de Anthropic |

El manager valida al arrancar que exista la key del proveedor seleccionado. Si falta, falla rápido con `LLMConfigError`.

## Cómo correrlo

```bash
python main.py
```

Esto hace la pregunta `"¿Qué es la entropía?"` dos veces:

1. **Modo normal**: espera la respuesta completa y muestra tokens usados + latencia.
2. **Modo streaming**: imprime los tokens a medida que llegan y mide el **TTFT** (Time-To-First-Token), que es la métrica clave de UX en productos LLM.

### Cambiar de proveedor

Solo tocás una línea en el `.env`:

```bash
# Antes
LLM_PROVIDER=openai

# Después
LLM_PROVIDER=anthropic
```

No cambia nada del código de aplicación. Ese es el punto del Factory + ABC.

## Uso programático

```python
import asyncio
from llm_client import AsyncLLMManager, ChatMessage, ModelConfig, Role

async def main():
    manager = AsyncLLMManager.from_env()

    messages = [ChatMessage(role=Role.USER, content="Hola")]
    config = ModelConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=256)

    # Modo normal
    response = await manager.generate(messages, config)
    print(response.content)

    # Streaming
    async for token in manager.stream(messages, config):
        print(token, end="", flush=True)

asyncio.run(main())
```

## Manejo de errores

Toda excepción del proveedor se normaliza a una subclase de `LLMError`:

| Excepción | Cuándo |
|---|---|
| `LLMConfigError` | Falta key o proveedor inválido |
| `LLMAuthError` | 401/403 del proveedor |
| `LLMRateLimitError` | 429 — candidato natural para retry con backoff |
| `LLMConnectionError` | Timeout, DNS, conexión caída |
| `LLMProviderError` | 5xx u otros errores de API |

El `main.py` las captura una por una y devuelve un exit code distinto según el tipo, para que un orquestador (cron, Airflow, lo que sea) pueda reaccionar sin parsear el stdout.

## Decisiones de diseño

- **ABC + Factory**: `BaseLLMClient` define el contrato; `AsyncLLMManager` es el punto único donde se decide qué proveedor instanciar. Cambiar de proveedor nunca toca la lógica de negocio.
- **SecretStr**: las API keys nunca aparecen en `repr`, `str` ni tracebacks de Pydantic. Para leer el valor hay que llamar explícitamente `.get_secret_value()`.
- **Normalización de respuesta**: ambos proveedores devuelven un `ModelResponse` con la misma forma. La app no necesita saber si vino `choices[0].message.content` (OpenAI) o `content[0].text` (Anthropic).
- **`system` prompt**: OpenAI lo trata como un mensaje más con `role=system`; Anthropic lo recibe como un parámetro aparte. El `ModelConfig.system_prompt` abstrae esa diferencia.
- **Fail-fast en la config**: el validator de Pydantic rechaza arrancar si falta la key del proveedor seleccionado, en vez de esperar al primer error de API en producción.

## Anti-patrones evitados

- ❌ Usar `OpenAI` (síncrono) dentro de `async def` — bloquea el event loop. Usamos `AsyncOpenAI`.
- ❌ Hardcodear la key en el código — usamos `.env` + `SecretStr`.
- ❌ Dejar que un `RateLimitError` crashee el proceso — lo normalizamos a `LLMRateLimitError` y lo capturamos.
- ❌ Instanciar clientes de SDK directamente en la lógica de negocio — todo va por el `AsyncLLMManager`.
