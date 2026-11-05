# Pre-entrega 2 — Refactorización a LCEL Asíncrono

Refactor del cliente manual de la pre-entrega 1 hacia una cadena **LCEL** (LangChain Expression Language) declarativa. El flujo principal se expresa con el operador `|` y se ejecuta de forma asíncrona con `.ainvoke()`.

## Objetivos cubiertos

| Criterio de aceptación | Dónde se resuelve |
|---|---|
| Sintaxis LCEL con `\|` para el flujo principal | `main.py` línea del `chain = prompt \| model \| parser` |
| Ejecución asíncrona (`await chain.ainvoke(...)`) | Dentro de `main()` |
| Salida en texto plano (StrOutputParser) | `parser = StrOutputParser()` |
| ChatPromptTemplate con roles System/Human | `prompt = ChatPromptTemplate.from_messages([...])` |
| Modelo LangChain (no SDK crudo) | `ChatAnthropic` / `ChatOpenAI` en `build_model()` |

## Estructura

```
pre-entrega-2/
├── main.py              # Ejercicio principal: cadena LCEL simple
├── bonus_parallel.py    # Bonus: RunnableParallel + encadenamiento
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Entorno virtual con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt

# 3. Configurar la API key
cp .env.example .env
# editar .env con tu key real
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | sí | `anthropic` (default) u `openai` |
| `ANTHROPIC_API_KEY` | si `LLM_PROVIDER=anthropic` | key de Anthropic |
| `OPENAI_API_KEY` | si `LLM_PROVIDER=openai` | key de OpenAI |

> **Nota**: no hace falta pasar la key al constructor de `ChatAnthropic` o `ChatOpenAI`; los wrappers de LangChain la leen automáticamente de la variable de entorno.

## Correrlo

### Ejercicio principal

```bash
python3.12 main.py
```

Hace una pregunta de física usando una cadena de 3 Runnables encadenados. Imprime la respuesta en texto plano y valida con un `assert` que la salida sea `str` (no `AIMessage`).

### Bonus: paralelismo declarativo

```bash
python3.12 bonus_parallel.py
```

Toma una reseña de cliente, en paralelo clasifica el sentimiento y extrae el tema, y después sintetiza una línea de ticket de soporte. Muestra en consola las piezas intermedias que fluyen entre los eslabones.

## Lo que demuestra cada archivo

### `main.py` — el patrón canónico de LCEL

```python
chain = prompt | model | parser
respuesta = await chain.ainvoke({"pregunta": "..."})
```

Tres objetos conectados por `|`. Cada uno implementa el protocolo **Runnable** (`invoke`, `ainvoke`, `batch`, `abatch`, `stream`, `astream`). La salida de uno se convierte en la entrada del siguiente:

```
dict -> [prompt] -> ChatPromptValue -> [model] -> AIMessage -> [parser] -> str
```

El factory `build_model()` permite alternar entre OpenAI y Anthropic sin tocar el resto de la cadena — misma lección que la pre-entrega 1, pero con una capa menos de código propio porque LangChain ya normaliza las diferencias entre proveedores.

### `bonus_parallel.py` — por qué LCEL vale la pena

Muestra lo que no se consigue con un SDK crudo sin escribir código de orquestación:

1. **`RunnableParallel`** dispara dos sub-cadenas en paralelo sobre la misma entrada. Su salida es un `dict` cuyas claves coinciden con las variables del siguiente prompt — así se encadena paralelo con secuencial sin código adicional.
2. **Tiempo total ≈ rama más lenta**, no la suma de las ramas. Equivalente a lo que hacías con `asyncio.gather` + `Semaphore` en la pre-entrega 1, pero declarativo.
3. **Debug directo**: podés invocar cualquier sub-cadena por sí sola para ver qué datos fluyen entre eslabones.

## Errores comunes a evitar (sacados en limpio del material)

- **Olvidar el `await`** antes de `chain.ainvoke(...)`. Sin él, obtenés una corrutina sin ejecutar y nada se imprime.
- **Variables del prompt que no matchean las del `dict`**. Si el prompt tiene `{pregunta}` y pasás `{"question": "..."}`, falla en runtime con un error poco amistoso.
- **Ignorar el output parser**. Sin `StrOutputParser()`, la salida es un `AIMessage` y hay que acceder a `.content` manualmente — rompiendo la cadena.
- **Usar `invoke` síncrono dentro de `async def`**. Siempre `ainvoke` / `abatch` / `astream` en código async, para no bloquear el event loop.

## Conexión con la pre-entrega 1

La pre-entrega 1 construía manualmente la abstracción que LCEL te da "gratis":

| Pre-entrega 1 (SDK crudo) | Pre-entrega 2 (LCEL) |
|---|---|
| `BaseLLMClient` ABC propio | protocolo `Runnable` de LangChain |
| `OpenAIClient` / `AnthropicClient` con `generate()` | `ChatOpenAI` / `ChatAnthropic` ya implementan `invoke`/`ainvoke` |
| `ModelResponse` normalizada con Pydantic | `AIMessage` normalizado + `StrOutputParser` |
| `AsyncLLMManager` como fachada | `chain = prompt \| model \| parser` |
| `asyncio.gather` manual para paralelo | `RunnableParallel` declarativo |

LCEL no reemplaza saber lo de la pre-entrega 1 — al revés, saberlo es lo que te permite debuggear LCEL cuando algo se rompe, porque vas a entender qué está haciendo la abstracción por dentro.
