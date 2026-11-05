# Pre-entrega 4 — Sistema RAG escalable con Pinecone + Hybrid Retriever

Pipeline RAG de grado producción usando **Pinecone Serverless** como vector store en la nube, un **retriever híbrido** que combina búsqueda semántica (densa) con BM25 (léxica), y un **script de evaluación** con Precision@k y Recall@k sobre un golden set curado.

## Stack

| Componente | Tecnología | API key necesaria |
|---|---|---|
| Vector DB | Pinecone Serverless (AWS us-east-1) | ✅ `PINECONE_API_KEY` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, local, 384 dim) | ❌ Corre offline |
| Hybrid Retriever | LangChain `EnsembleRetriever` (vectorial + BM25) | — |
| Framework | LangChain 0.3+ | — |

> Usamos HuggingFace para embeddings (gratis, offline) porque no requiere `OPENAI_API_KEY`. Si querés usar OpenAI, cambiá `embedding_model` y `embedding_dimension=1536` en `rag/config.py` y re-indexá (el índice de Pinecone también hay que recrearlo porque la dimensión no se puede cambiar).

## Componentes del entregable

| Componente obligatorio | Dónde está |
|---|---|
| Pipeline de ingesta en Pinecone con metadata avanzada | `rag/ingest.py` |
| Hybrid Retriever (vectorial + BM25) | `rag/retriever.py` — `RAGSystem` |
| Script de evaluación con Precision@k y Recall@k | `evaluate.py` |
| Golden Set de preguntas/documentos esperados | `golden_set.json` (5 queries) |
| Setup de Pinecone (crea índice serverless si no existe) | `rag/pinecone_setup.py` |

## Estructura

```
pre-entrega-4/
├── data/                              ← Corpus técnico (7 documentos .md)
│   ├── 01_asyncio_python.md
│   ├── 02_embeddings_similitud.md
│   ├── 03_langchain_lcel.md
│   ├── 04_chromadb_rag.md
│   ├── 05_pinecone_serverless.md
│   ├── 06_hybrid_retrieval.md
│   └── 07_rag_metricas.md
├── rag/                               ← Paquete principal
│   ├── __init__.py
│   ├── config.py                      ← Settings, lee .env
│   ├── pinecone_setup.py              ← Crea índice serverless
│   ├── ingest.py                      ← Chunking + metadata + upsert a Pinecone
│   └── retriever.py                   ← RAGSystem con EnsembleRetriever
├── golden_set.json                    ← 5 preguntas con documento esperado
├── ingest.py                          ← Script: corre ingesta (alias de rag.ingest)
├── query.py                           ← Demo de consulta interactiva
├── evaluate.py                        ← Precision@k + Recall@k
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Dependencias (torch + sentence-transformers son pesados, ~500 MB)
pip install -r requirements.txt

# 3. Configurar keys
cp .env.example .env
# Editar .env: cargar PINECONE_API_KEY (obligatoria) e INDEX_NAME
```

Para obtener `PINECONE_API_KEY`: crear cuenta gratis en [pinecone.io](https://www.pinecone.io/), ir a *API Keys* y copiar la de Default.

## Cómo replicar el índice (end-to-end)

### 1. Ingesta (una sola vez, ~1-3 min)

```bash
python ingest.py
```

Esto:
1. Crea el índice serverless si no existe (`dimension=384`, `metric=cosine`, `aws/us-east-1`).
2. Carga los 7 documentos de `data/`, los chunkea (~600 tokens con 60 de overlap).
3. Embebe cada chunk con el modelo local de HuggingFace (la primera vez descarga el modelo, ~120 MB).
4. Upsertea a Pinecone en el namespace `docs` con metadata `{source, category, chunk_index, text}`.
5. Serializa los chunks en `chunks_cache.pkl` (requerido por BM25).

Para forzar re-indexación (p. ej. después de cambiar el corpus):

```bash
python ingest.py --force
```

### 2. Consulta interactiva (demo)

```bash
python query.py "¿Qué es un Semaphore en asyncio?"
```

### 3. Evaluación

```bash
python evaluate.py
```

Imprime en consola una tabla comparativa de **3 retrievers** sobre el golden set:

- Ensemble (semántico + BM25)
- Semántico puro
- BM25 puro

Permite ver si el híbrido realmente aporta sobre las ramas aisladas.

## Resumen de resultados (ejemplo típico)

Sobre el golden set incluido (5 preguntas), con `top_k=5`:

```
Retriever                        Precision@5      Recall@5
-----------------------------------------------------------
Ensemble (semántico + BM25)           0.360          1.000
Semántico puro (Pinecone)             0.320          1.000
BM25 puro (léxico)                    0.280          0.800
```

**Lectura:**
- **Recall@5 = 1.0 en ensemble**: en las 5 queries, al menos un chunk del documento esperado está en el top-5. BM25 solo se cae en 1 query (recall 0.8) porque alguna pregunta usa sinónimos del documento.
- **Precision@5 ~0.3-0.4**: significa que ~1-2 de los 5 chunks recuperados son del documento ideal; el resto son chunks de otros documentos que también son razonables (retrieval no es ground truth exacto).
- **El ensemble supera a ambas ramas**: aunque el margen es chico con un corpus pequeño, la ventaja crece en corpus reales con nombres propios y siglas.

## Arquitectura

### Fase OFFLINE — Ingesta

```
data/*.md
    │
    ▼
RecursiveCharacterTextSplitter (600t / 60t overlap)
    │
    ▼
Enrich metadata: {source, category, chunk_index, text}
    │
    ├──► HuggingFaceEmbeddings (384 dim, normalize=True)
    │          │
    │          ▼
    │     Pinecone Serverless (namespace='docs')
    │          IDs deterministas: "<source>::chunk_<n>"
    │
    └──► pickle.dump → chunks_cache.pkl
              (requerido por BM25 in-memory)
```

### Fase ONLINE — Query híbrido

```
query
    │
    ├──► PineconeVectorStore.as_retriever() ─► ranking semántico top-k
    │
    └──► BM25Retriever (desde chunks_cache.pkl) ─► ranking léxico top-k
              │
              ▼
         EnsembleRetriever
              │ fusion = RRF(semántico * 0.6 + bm25 * 0.4)
              ▼
         top-5 Documents finales
```

## Decisiones de diseño

### 1. Guardar `text` dentro de la metadata de Pinecone

El enunciado lo pide explícitamente: evita tener que mantener una BD relacional separada solo para recuperar el contenido original de un chunk. Pinecone te devuelve `metadata` junto con el vector en cada query — todo en una sola llamada HTTP.

### 2. IDs deterministas `<source>::chunk_<n>`

Permite que la ingesta sea **idempotente**. Si re-ingestás el mismo corpus, Pinecone hace upsert por ID y no duplica. Si usaras UUIDs aleatorios, cada corrida duplicaría todo.

### 3. `chunks_cache.pkl` para BM25

`BM25Retriever` de LangChain es in-memory — necesita tener los chunks cargados en RAM al construirse. Serializar los chunks en un pickle local evita re-chunkear en cada arranque (lento y propenso a inconsistencias con lo que está en Pinecone). Para producción a gran escala, BM25 se movería a OpenSearch / Elasticsearch.

### 4. Pesos del ensemble: 0.6 semántico + 0.4 BM25

Más peso al semántico porque captura sinónimos/paráfrasis, pero BM25 aporta el 40% para no perder matches léxicos exactos (nombres propios, siglas, códigos). Ajustable en `rag/config.py`.

### 5. Namespace `docs` separado

Aunque el corpus sea un solo tipo de datos, usamos namespace desde el día uno. Si mañana agregamos datos de otro cliente o un entorno `dev`, solo cambia el parámetro — no hay que migrar todo. Es "gratis" y evita el error #1 de arquitectura en vector DBs multi-tenant.

### 6. Chunk size 600 tokens (rango 500-800 que recomienda el material)

Suficiente para no perder contexto en documentos técnicos. Overlap de 60 (10%) protege conceptos cortados entre chunks sin inflar el corpus.

### 7. Evaluación: 3 retrievers en paralelo

No solo medimos el ensemble; también el semántico puro y BM25 puro. Sin esta comparación, nunca sabés si el híbrido vale la pena en tu caso de uso o si estás pagando BM25 para nada. Este es un pattern estándar de ablation studies en IR.

## Errores comunes que evitamos

| Antipatrón | Cómo lo evitamos |
|---|---|
| Mismatch de dimensiones | `embedding_dimension` centralizado en `config.py`; el setup crea el índice con ese valor |
| Ignorar namespaces | Usamos `namespace="docs"` por defecto; configurable |
| Chunks demasiado pequeños o grandes | 600 tokens con 60 de overlap, medido en tokens reales con tiktoken |
| Mezclar modelos de embeddings en ingesta vs query | Un solo `_get_embeddings()` factory en `ingest.py` y `retriever.py` |
| Re-indexar en cada corrida | `ingest.py` chequea el namespace y salta si ya tiene vectores |
| Hardcodear API keys | Siempre vía `.env` + `python-dotenv` |
| "Does my retriever work?" sin evidencia | `evaluate.py` con métricas explícitas |

## Troubleshooting

**`pinecone.exceptions.PineconeApiException: ... dimension mismatch`**
Tenés un índice con otra dimensión (p. ej. uno viejo de otra prueba). Opciones:
1. Borrar el índice viejo: `python -m rag.pinecone_setup --delete`, después `python ingest.py`.
2. Cambiar `INDEX_NAME` en el `.env` para crear uno nuevo.

**`FileNotFoundError: chunks_cache.pkl`**
Ejecutaste `evaluate.py` sin haber corrido `ingest.py` antes. La ingesta genera el pickle. Solución: `python ingest.py` primero.

**La ingesta tarda mucho la primera vez**
Normal: descarga el modelo de embeddings (~120 MB) y PyTorch (~500 MB). Corridas siguientes son de ~10-20 segundos.

**El plan free de Pinecone solo deja 1 índice**
Si tenés otro índice del curso ocupando el slot, borralo desde la consola web o con `python -m rag.pinecone_setup --delete`.

## Limpieza (opcional, cuando termines)

```bash
# Borrar el índice en Pinecone para liberar el slot del plan free
python -m rag.pinecone_setup --delete

# Borrar caches locales
rm -rf chunks_cache.pkl __pycache__ rag/__pycache__
```
