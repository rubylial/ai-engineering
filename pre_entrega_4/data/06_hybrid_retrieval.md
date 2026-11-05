# Hybrid Retrieval: combinando búsqueda léxica y semántica

## El problema de la búsqueda puramente semántica

Los embeddings densos capturan significado pero fallan en algunos escenarios
críticos:

- **Nombres propios y códigos**: si buscás "modelo gpt-4o-mini-2024-07-18",
  un embedding puede devolverte menciones genéricas de GPT-4 que están
  semánticamente cerca pero no son el modelo exacto.
- **Términos técnicos poco frecuentes**: palabras como "DynamoDB", "RAFT",
  "OAuth2" tienen poca representación en el embedding promedio, y el modelo
  puede confundirlas con conceptos vecinos.
- **Siglas y acrónimos**: "RAG", "ETL", "CRUD" suelen quedar diluidos en
  el espacio vectorial frente a términos más genéricos.

## BM25: la contraparte léxica

BM25 (Best Matching 25) es un algoritmo de scoring basado en frecuencia de
términos, inversamente ponderado por la frecuencia en todo el corpus. Es
determinista, rápido y especialmente efectivo cuando la query contiene
términos exactos que deben aparecer en el documento.

BM25 no entiende sinónimos ni paráfrasis: si el usuario busca "velocidad"
pero el documento dice "rapidez", BM25 no encuentra nada.

## Ensemble: lo mejor de ambos mundos

Un retriever híbrido combina ambos enfoques:

- **Búsqueda semántica (densa)**: embeddings + vector database. Captura
  sinónimos, paráfrasis y significado.
- **Búsqueda léxica (sparse)**: BM25. Captura coincidencias exactas,
  términos técnicos y nombres propios.

Los scores de cada retriever se combinan con pesos ajustables (p. ej. 0.6
semántico + 0.4 léxico). LangChain provee `EnsembleRetriever` para esto.

## Reciprocal Rank Fusion (RRF)

La estrategia más común para fusionar rankings de dos retrievers es RRF:
cada documento recibe una puntuación de 1/(k + rank), donde rank es su
posición en cada ranking individual y k es un hiperparámetro (típicamente 60).
Los scores se suman entre retrievers y los documentos se re-ordenan.

LangChain usa RRF por defecto en EnsembleRetriever.

## Limitaciones del enfoque in-memory

BM25Retriever de LangChain mantiene el índice léxico en RAM. Para corpus
pequeños (miles de docs) funciona perfectamente. Para millones de documentos,
hay que mover BM25 a OpenSearch o Elasticsearch, o usar índices sparse
nativos en Pinecone (hybrid search con vectores sparse).
