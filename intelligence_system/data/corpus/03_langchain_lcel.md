---
doc_id: 03_langchain_lcel
title: "LangChain Expression Language (LCEL)"
category: langchain
tags: [lcel, runnable]
source: 03_langchain_lcel.md
---

# LangChain Expression Language (LCEL)

## Qué es LCEL

LCEL es un lenguaje declarativo para componer cadenas de componentes de IA.
Su sintaxis característica es el operador pipe (`|`), que conecta la salida de
un componente con la entrada del siguiente.

## El protocolo Runnable

Todos los componentes de LangChain (prompts, modelos, parsers, retrievers,
herramientas) implementan la clase base Runnable. Esto garantiza que todos
expongan los mismos métodos: invoke, ainvoke, batch, abatch, stream, astream.

Esta interfaz uniforme es lo que habilita la composición con pipe: como todos
los Runnables hablan el mismo idioma, pueden encadenarse sin código de pegamento.

## Estructura típica de una cadena RAG

Un flujo RAG completo suele verse así:

```
retriever | format_docs | prompt | model | parser
```

El retriever busca documentos relevantes, format_docs los convierte en un string
plano para inyectar en el prompt, el prompt arma el mensaje final, el modelo
genera la respuesta y el parser la convierte en el formato deseado.

## RunnableParallel

Dispara varias sub-cadenas simultáneamente sobre la misma entrada. El tiempo
total es el de la rama más lenta, no la suma. Esencial cuando se necesitan
hacer varias consultas independientes al modelo.

Además de concurrencia, sirve para mapear la entrada: produce un dict cuyas
claves alimentan directamente las variables del siguiente prompt.

## ChatPromptTemplate

Permite definir prompts con roles (system, human, assistant). El rol system es
un espacio protegido donde el modelo toma instrucciones de comportamiento con
más peso que las del turno del usuario. Es crucial para RAG: ahí se indica
"responde solo con el contexto provisto" o "no inventes información".

## Modo asíncrono

En aplicaciones web (FastAPI, Lambda) siempre debe preferirse `ainvoke`, `abatch`,
`astream` sobre las versiones sincrónicas. Esto evita que el event loop se
bloquee mientras se espera la respuesta de la API externa, permitiendo que el
servidor atienda otras requests en paralelo.

Olvidarse del `await` antes de `ainvoke` es el bug número uno al empezar con
LCEL: en vez del resultado se obtiene una corrutina sin ejecutar.

## Salidas estructuradas

Para forzar que el modelo devuelva un JSON con estructura fija, LangChain ofrece
dos mecanismos:

- `PydanticOutputParser`: inyecta las instrucciones de formato en el prompt y
  parsea el string JSON de respuesta. Compatible con cualquier modelo.
- `with_structured_output(Schema)`: usa tool-calling nativo del proveedor (más
  robusto, no depende de que el modelo siga instrucciones de texto).

## Resiliencia con with_retry

`chain.with_retry(stop_after_attempt=3)` reintenta automáticamente ante fallos
transitorios (rate limits, timeouts de red, 5xx del proveedor). No reintenta
errores lógicos como ValidationError, porque reintentarlos solo gasta créditos.
