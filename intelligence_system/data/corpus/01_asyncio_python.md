---
doc_id: 01_asyncio_python
title: "Asyncio en Python 3.12 para ingeniería de IA"
category: python
tags: [asyncio, concurrencia]
source: 01_asyncio_python.md
---

# Asyncio en Python 3.12 para ingeniería de IA

## Por qué asyncio

Las llamadas a LLMs, bases vectoriales y APIs externas son operaciones I/O-bound.
En un modelo síncrono, el programa queda bloqueado esperando la respuesta del
servidor sin poder atender a nadie más. Asyncio permite que un solo proceso
gestione múltiples tareas concurrentemente mediante concurrencia cooperativa:
mientras una tarea espera datos de la red, cede el control a otra que puede
avanzar.

Asyncio NO acelera el cálculo matemático. Para tareas CPU-bound (entrenar un
modelo, hacer feature engineering pesado con pandas) se necesita multiprocessing.
Asyncio solo ayuda en tareas I/O-bound.

## Primitivas fundamentales

### Corrutinas (async def)

Una corrutina es una función que puede pausar su ejecución con `await` para ceder
el control al event loop. Hasta que no se aplica `await` sobre ella, no hace nada.

### El Event Loop

Es el motor invisible que coordina las corrutinas. Recibe tareas, las asigna y
cuando una está esperando I/O, le da espacio a otra para avanzar.

### Tasks

Una task es una corrutina programada para ejecutarse lo antes posible. A diferencia
de una corrutina suelta, una task empieza a correr en segundo plano mientras el
código continúa.

## Punto de entrada: asyncio.run()

A partir de Python 3.12 nunca se debe usar `asyncio.get_event_loop()` directamente.
`asyncio.run(main())` es el estándar: gestiona la creación del loop, la ejecución
de las tareas y el cierre limpio.

## Patrones de concurrencia

### asyncio.gather

Lanza múltiples corrutinas en paralelo y espera a que todas terminen. Se usa
cuando necesitás comparar respuestas de varios modelos al mismo tiempo. El
parámetro `return_exceptions=True` es crucial: hace que un fallo individual
no tumbe todo el lote.

### asyncio.Semaphore

Un semáforo limita cuántas corrutinas pueden ejecutarse concurrentemente. Es
indispensable cuando trabajas contra APIs con rate limits. Por ejemplo, para
embeddear 10.000 textos, NUNCA se usa `asyncio.gather` sobre las 10.000 llamadas:
se encolan todas pero un semáforo restringe a 5-10 llamadas en paralelo, evitando
el error 429 (Too Many Requests).

### asyncio.timeout

Nunca dejes una llamada a un LLM abierta indefinidamente. `async with asyncio.timeout(10.0)`
corta la operación si tarda más del umbral. Captura la excepción `TimeoutError`
para aplicar una estrategia de fallback.

## Antipatrones

El más grave es el "bloqueo del loop": llamar a `time.sleep(5)` o a una función
síncrona pesada desde dentro de una corrutina. Esto congela el event loop completo
y todos los usuarios quedan colgados. La solución es `await asyncio.sleep(5)` o,
si debes usar una librería síncrona antigua, envolverla con `asyncio.to_thread()`.
