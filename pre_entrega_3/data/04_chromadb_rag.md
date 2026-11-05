# ChromaDB para RAG

## Por qué ChromaDB en local

ChromaDB es una base de datos vectorial open source diseñada para ser simple de
operar. En modo PersistentClient guarda los datos en archivos .parquet y .bin
en una carpeta del disco local, sin requerir levantar un contenedor Docker.

Es ideal para desarrollo, prototipos y aplicaciones privadas de escritorio.
Para producción a gran escala se recomiendan alternativas como Pinecone,
Weaviate o Qdrant que ofrecen mejor escalabilidad horizontal.

## Operaciones CRUD en una vector database

### Create / Upsert

No solo se guarda el texto sino su vector asociado. La operación upsert es
idempotente: si el ID existe se actualiza; si no, se crea. Es el patrón correcto
para pipelines de ingesta que corren múltiples veces sobre el mismo corpus.

### Read: get vs query

- `get()` busca por IDs exactos o por filtros de metadata — análogo a un SELECT
  tradicional.
- `query()` busca por similitud semántica. Es el corazón de RAG.

### Update y Delete

Update modifica el documento o sus metadatos (re-disparando la vectorización).
Delete elimina vectores obsoletos, importante para mantener el conocimiento
fresco del agente.

## El rol de los metadatos

Los metadatos son etiquetas adjuntas a cada vector, por ejemplo:
`{"source": "manual_v2.pdf", "autor": "Juan", "fecha": "2024-05"}`.

Son cruciales para dos cosas:

1. **Filtrar antes de la búsqueda vectorial** (pre-filtering). ChromaDB
   acepta un parámetro `where` en `query()` que restringe la búsqueda a los
   documentos que cumplen el filtro, antes de calcular similitudes. Esto es
   mucho más eficiente que buscar top-N y filtrar después.
2. **Rastrear el origen** de la información para citar fuentes en la respuesta
   del LLM.

## Errores comunes

### ID no determinista

Usar UUIDs aleatorios hace imposible el upsert eficiente. Se recomienda usar
hashes del contenido (como SHA-256 del texto) o IDs estructurados
(`manual_v2__section_3__chunk_7`).

### Desajuste de embeddings

Intentar consultar una colección con un modelo de embeddings distinto al que
se usó para indexarla produce resultados aleatorios. El modelo de indexación
y el de consulta deben ser exactamente el mismo.

### Falta de persistencia

Inicializar `chromadb.Client()` (en memoria) en lugar de `PersistentClient(path=...)`
causa pérdida total de datos al cerrar el script. Siempre persistir en disco
para corpus que se reutilizan.

## Chunking antes de ingestar

Los documentos grandes deben trocearse antes de embedder. Chunks de 500 tokens
con 50 de overlap es un punto de partida sensato. Menos de 200 tokens pierde
contexto; más de 1000 diluye el significado en el vector promedio.

El overlap garantiza que un concepto cortado a la mitad entre chunks aparezca
completo en al menos uno de ellos.

## Integración con LangChain

LangChain provee la clase `Chroma` que envuelve ChromaDB como un VectorStore
compatible con el protocolo Runnable. Esto permite usarlo directamente en
cadenas LCEL con `.as_retriever()`, componiéndolo con pipes.
