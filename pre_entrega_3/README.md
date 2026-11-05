# Pre-entrega 3 — Sistema RAG End-to-End

Implementación de un pipeline **RAG (Retrieval-Augmented Generation)** completo con LangChain, ChromaDB, **Claude (Anthropic)** y **embeddings locales de HuggingFace**. El sistema responde preguntas basándose *exclusivamente* en un corpus de documentos locales, y rechaza preguntas cuya respuesta no esté en el contexto (anti-alucinación).

## Stack

| Componente | Tecnología | API key necesaria |
|---|---|---|
| LLM | `claude-haiku-4-5` (Anthropic) | ✅ `ANTHROPIC_API_KEY` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, local) | ❌ Corre offline |
| Vector store | ChromaDB persistente | ❌ Local |
| Framework | LangChain LCEL (async) | — |

> **Por qué HuggingFace local para embeddings**: Anthropic no ofrece una API propia de embeddings, así que usamos un modelo multilingual de sentence-transformers que se descarga una vez (~120 MB) y después corre offline y gratis. Es ideal para corpus pequeños/medianos y evita tener que contratar otra API.

## Objetivos cubiertos

| Entregable | Dónde se resuelve |
|---|---|
| Script de ingesta con chunking (500 tokens, 50 overlap) + ChromaDB | `rag/ingest.py` |
| Cadena RAG asíncrona (LCEL) con retriever + prompt + LLM + parser | `rag/chain.py` |
| Dataset de ejemplo | `data/*.md` (4 archivos sobre asyncio, embeddings, LCEL, ChromaDB) |
| Prompt grounded ("responde solo con el contexto") | `SYSTEM_PROMPT` en `rag/chain.py` |
| Salida estructurada con `PydanticOutputParser` | `rag/schemas.py` + cadena en `chain.py` |
| Función async `get_rag_response(query)` | `rag/chain.py` — API pública |
| Dos pruebas: una respondible y una trampa | `main.py` |
| Persistencia que NO re-indexa si ya existe | `ingest.py` — check de `_collection.count()` |
| API key en `.env` (nunca hardcodeada) | `config.py` + `.env.example` |

## Estructura

```
pre-entrega-3/
├── data/                          ← Dataset de ejemplo (editable)
│   ├── 01_asyncio_python.md
│   ├── 02_embeddings_similitud.md
│   ├── 03_langchain_lcel.md
│   └── 04_chromadb_rag.md
├── rag/                           ← Paquete principal
│   ├── __init__.py
│   ├── config.py                  ← Settings (lee .env)
│   ├── schemas.py                 ← RAGResponse + RAGReference
│   ├── ingest.py                  ← Chunking + indexación en Chroma (HuggingFace embeds)
│   └── chain.py                   ← Cadena LCEL asíncrona (Claude + PydanticOutputParser)
├── main.py                        ← Entry point: ingesta + 2 pruebas
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Requiere **Python 3.12+**.

```bash
# 1. Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Dependencias (instala sentence-transformers que es pesado, ~500 MB con torch)
pip install -r requirements.txt

# 3. API key
cp .env.example .env
# editar .env con tu ANTHROPIC_API_KEY real
```

## Cómo correrlo

### Opción A: todo en un paso (recomendado para evaluar la entrega)

```bash
python main.py
```

Esto:
1. Descarga el modelo de embeddings (la primera vez, ~120 MB, tarda 1-2 min).
2. Indexa el corpus en Chroma (si no está indexado).
3. Corre las dos pruebas y valida que el sistema no alucine.

### Opción B: en pasos separados

```bash
# Solo ingesta
python -m rag.ingest

# Forzar re-indexación (borrar vectorstore anterior)
python -m rag.ingest --force
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `ANTHROPIC_API_KEY` | sí | Se usa solo para el LLM. Los embeddings corren locales. |

## Arquitectura

### Fase OFFLINE — Ingesta (única vez o cuando cambia el corpus)

```
data/*.md
    │
    ▼
Cargar documentos (langchain Document con metadata)
    │
    ▼
RecursiveCharacterTextSplitter
  · chunk_size=500 tokens  · chunk_overlap=50 tokens
  · length_function mide en TOKENS (tiktoken cl100k_base)
  · separators: ["\n\n", "\n", ". ", " ", ""]
    │
    ▼
HuggingFaceEmbeddings (local, offline)
  · paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
  · normalize_embeddings=True → norma 1, apto para coseno
    │
    ▼
ChromaDB PersistentClient (./vectorstore/)
  · IDs deterministas: "<source>::chunk_<n>"
  · Metadata: {"source": "<nombre_archivo>"}
```

### Fase ONLINE — Query (por cada pregunta)

```
pregunta
    │
    ├──► retriever.ainvoke(pregunta)   [top-k=4 por similitud]
    │          │
    │          ▼
    │     [Document, ...]
    │          │
    │          ▼
    │     _format_docs → string con marcas [source: X.md]
    │
    └──► RunnableParallel({context, question})
              │
              ▼
         ChatPromptTemplate (system + human + format_instructions)
              │
              ▼
         ChatAnthropic (claude-haiku-4-5)
              │
              ▼
         PydanticOutputParser → RAGResponse
```

## Decisiones de diseño

### 1. El prompt como "filtro de veracidad"

El `SYSTEM_PROMPT` en `rag/chain.py` tiene 5 reglas numeradas que el modelo debe seguir. Lo esencial:

- *"Si la respuesta NO está en el contexto, `answer` debe decir: 'No tengo acceso a esa información en mi base de conocimiento.'"*
- *"NUNCA inventes información ni completes con conocimiento propio."*
- *"Devolvé EXCLUSIVAMENTE un JSON válido que cumpla el esquema."*

Esto se valida en `main.py` con la prueba trampa.

### 2. Sin `temperature` para Claude

El SDK de Anthropic **2.5.0+** removió `temperature` y `top_p` de `messages.create()` — pasarlos tira `"Extra inputs are not permitted"`. El código no los pasa. Para RAG esto no es un problema: Claude por defecto ya responde de forma bastante determinista.

### 3. `top_k=4` (no más, no menos)

- **Lost in the Middle**: con más de 5 fragmentos, el modelo pierde atención sobre los del medio.
- **Costo**: cada fragmento adicional son ~500 tokens extra en el prompt → más latencia.

### 4. IDs deterministas: `<source>::chunk_<n>`

Permiten upserts idempotentes. Si re-ingestamos el mismo archivo, Chroma reemplaza los chunks con esos IDs en lugar de duplicarlos.

### 5. `PydanticOutputParser` vs `with_structured_output`

El enunciado pide **`PydanticOutputParser`** explícitamente, así que lo usamos. Inyecta las instrucciones de formato en el prompt y parsea el JSON de respuesta — compatible con cualquier LLM, no depende de tool-calling nativo del proveedor.

### 6. `with_retry(stop_after_attempt=3)` al final de la cadena

Resiliencia ante rate limits (429), timeouts de red y 5xx de Anthropic. No reintenta errores lógicos como `ValidationError`.

### 7. Embeddings normalizados (norma 1)

El encoder los normaliza con `normalize_embeddings=True`. Combinado con la métrica L2 por defecto de Chroma, el resultado es equivalente a similitud coseno pero más barato de computar.

### 8. Chunking medido en **tokens**, no en caracteres

Aunque el LLM final es Claude, medimos con `tiktoken.cl100k_base` como presupuesto consistente. Si en el futuro cambiás el LLM, el chunking sigue siendo sensato.

## Pruebas incluidas en `main.py`

### Prueba 1 — pregunta cuya respuesta está en el corpus

```
❯ ¿Para qué sirve asyncio.Semaphore y cuándo lo usarías?
```

**Salida esperada:**
- `has_answer: True`
- `answer`: explicación basada en `01_asyncio_python.md`
- `references`: al menos una, apuntando a ese archivo

### Prueba 2 — pregunta trampa

```
❯ ¿Cuál es el precio actual de Bitcoin en pesos colombianos?
```

**Salida esperada:**
- `has_answer: False`
- `answer`: *"No tengo acceso a esa información en mi base de conocimiento."*
- `references`: `[]`

Si el modelo inventara un precio, `main.py` falla con `AssertionError` ("ALUCINACIÓN detectada").

## Errores comunes que ya evita este diseño

| Antipatrón | Cómo lo evitamos |
|---|---|
| Contexto infinito (> 5 docs al LLM) | `top_k=4` explícito |
| Mezclar modelos de embeddings | `config.Settings` centraliza el modelo; mismo en ingest y en query |
| Re-indexar en cada corrida | `ingest_data_directory()` chequea `_collection.count() > 0` |
| API keys en código | Siempre vía `.env` + `python-dotenv` |
| Alucinación silenciosa | `has_answer: bool` en el schema + assert en `main.py` |
| Bloqueo del event loop | `ainvoke` en todo; API pública es `async def get_rag_response(...)` |

## Troubleshooting

**La primera corrida tarda bastante**
Normal: la primera vez descarga el modelo de embeddings (~120 MB) y algunas dependencias de torch. Después queda cacheado en `~/.cache/huggingface/` y las corridas siguientes son de ~3-5 segundos.

**Error: `No se encontraron archivos .md/.txt en ./data`**
Verificá que estés corriendo desde el directorio raíz del repo, no desde dentro de `rag/`.

**Error "Extra inputs are not permitted" al llamar a Claude**
Es porque alguien agregó `temperature=` o `top_p=` en `_build_llm()`. En el SDK de anthropic ≥ 2.5.0 esos parámetros fueron removidos. El código provisto ya no los pasa.

**Quiero cambiar el corpus**
Reemplazá archivos en `data/`, borrá `vectorstore/` (`rm -rf vectorstore/`), y volvé a correr `python main.py`. También podés usar `python -m rag.ingest --force`.

**Quiero usar sentence-transformers en GPU**
Editá `rag/ingest.py`, cambiá `model_kwargs={"device": "cpu"}` por `{"device": "cuda"}`. Requiere PyTorch con soporte CUDA.
