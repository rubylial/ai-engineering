---
doc_id: 02_embeddings_similitud
title: "Embeddings y similitud semántica"
category: embeddings
tags: [embeddings, similitud]
source: 02_embeddings_similitud.md
---

# Embeddings y similitud semántica

## Qué es un embedding

Un embedding es una representación numérica densa de un objeto (típicamente texto)
en un espacio vectorial de alta dimensionalidad. Es una lista de números (un vector)
que captura el "significado" de un fragmento de texto.

Antes de los embeddings modernos se usaba One-Hot Encoding, donde cada palabra era
un vector gigante lleno de ceros con un solo "1" en una posición. En ese sistema,
la distancia matemática entre "perro" y "canino" era la misma que entre "perro" y
"microprocesador". No había noción de significado.

Los embeddings modernos son vectores densos de 1536 a 3072 dimensiones. En ese
espacio, cada dimensión representa una característica abstracta aprendida por el
modelo durante su entrenamiento. "Rey" queda cerca de "Hombre" y "Reina" cerca de
"Mujer", manteniendo distancias lógicas entre conceptos.

## Métricas de similitud

### Similitud coseno

Mide el coseno del ángulo entre dos vectores. Captura la dirección (de qué tema
habla el texto) e ignora la magnitud (longitud). Es el estándar en RAG porque
dos textos del mismo tema pero de distinta longitud quedan correctamente
identificados como similares. Rango: típicamente de 0 a 1 en embeddings de texto.

### Distancia euclídea (L2)

Mide la distancia en línea recta entre dos puntos. Es muy sensible a la magnitud,
lo que la vuelve problemática para comparar textos de diferente longitud. Útil
solo cuando los vectores fueron normalizados previamente.

### Producto punto (dot product)

Multiplica magnitudes y coseno del ángulo. Si los vectores están normalizados
(norma 1), dot product equivale a coseno pero es más barato de computar. Por eso
muchas vector databases piden normalizar y usar dot.

## Modelos contextuales vs estáticos

Los modelos antiguos (Word2Vec, GloVe) asignaban un único vector a cada palabra:
"banco" (de sentarse) y "banco" (entidad financiera) compartían embedding. Los
modelos modernos (BERT, text-embedding-3, Voyage, Cohere) son contextuales: la
misma palabra genera vectores distintos según las palabras que la rodean.

## Errores comunes

### Negación invisible

Los embeddings capturan temática, no lógica. Textos como "hay stock" y "no hay
stock" pueden quedar peligrosamente cerca en el espacio vectorial porque ambos
hablan de disponibilidad.

### Mezclar modelos

NUNCA compares un embedding de OpenAI con uno de HuggingFace. Viven en espacios
vectoriales distintos y con dimensiones diferentes. No hay relación matemática
entre ellos. Si cambias de modelo de embedding en producción, tenés que
re-embedder todo el corpus.

### Domain shift

Un modelo entrenado con textos generales de internet puede fallar en lenguaje
médico muy especializado o en código fuente técnico. Para esos dominios se
recomienda fine-tuning o modelos específicos (BioBERT, SapBERT).

### Límite de tokens

Cada modelo de embedding tiene una ventana de contexto. Intentar embedder un PDF
de 100 páginas en un solo vector trunca o diluye el significado. Por eso existe
el chunking.
