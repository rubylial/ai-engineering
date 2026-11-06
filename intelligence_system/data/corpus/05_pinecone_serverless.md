---
doc_id: 05_pinecone_serverless
title: "Pinecone Serverless para infraestructura RAG"
category: vector_db
tags: [pinecone, serverless]
source: 05_pinecone_serverless.md
---

# Pinecone Serverless para infraestructura RAG

## Por qué Pinecone en la nube

Pinecone es una base de datos vectorial nativa de la nube, diseñada para
manejar millones o miles de millones de vectores con latencias de milisegundos.
A diferencia de bases locales como ChromaDB, Pinecone ofrece:

1. **Arquitectura serverless**: no requiere gestionar nodos ni memoria. Pagás
   por lo que usás y el almacenamiento escala automáticamente.
2. **Separación de cómputo y almacenamiento**: las búsquedas son rápidas sin
   importar cuántos datos haya almacenados.
3. **Persistencia garantizada**: los datos viven en la infraestructura
   administrada de Pinecone, no en tu contenedor local.

## Conceptos fundamentales

### Index

Es el contenedor principal. Define la dimensión de los vectores (que debe
coincidir exactamente con el modelo de embeddings) y la métrica de similitud:
cosine, dotproduct o euclidean.

### Namespace

Permite segmentar un mismo índice en múltiples particiones lógicas. Es crucial
para implementar multi-tenancy (aislar datos de distintos clientes) o separar
entornos (dev, staging, prod) dentro de un mismo índice, sin pagar por múltiples
índices.

### Metadata

Información adjunta al vector (JSON) que permite realizar filtrados adicionales
durante la búsqueda vectorial. Los filtros usan una sintaxis SQL-like con
operadores como `$eq`, `$in`, `$gt`, `$and`, `$or`.

## Pipeline típico

En un entorno profesional, Pinecone se integra en un pipeline:

1. **Ingesta**: los documentos se fragmentan (chunking), se convierten a
   vectores (embedding) y se suben con `upsert` a Pinecone.
2. **Retrieval**: la consulta del usuario se vectoriza, se busca el vecino más
   cercano en Pinecone filtrando por metadatos, y el resultado alimenta al LLM.

## Errores comunes de arquitectura

### Mismatch de dimensiones

Intentar insertar un vector de 1536 dimensiones en un índice configurado para
768 lanza un error inmediato. Las dimensiones se definen al crear el índice y
no se pueden cambiar después: hay que crear uno nuevo.

### Ignorar los namespaces

Mezclar datos de prueba con producción en el mismo namespace compromete la
seguridad y la claridad de los resultados. Es gratis usar namespaces, no hay
razón para no hacerlo.

### Elección errónea de métrica

Usar similitud euclidiana cuando el modelo de embeddings fue entrenado
específicamente para optimizar coseno produce resultados degradados. Si los
vectores están normalizados (norma 1), dot product es equivalente a coseno
pero más barato de computar.
