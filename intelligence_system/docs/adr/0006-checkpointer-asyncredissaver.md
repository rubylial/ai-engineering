# ADR-0006 · Persistencia con AsyncRedisSaver sobre Redis 8

- Estado: aceptada · Fecha: 2026-11-05

## Contexto
Las conversaciones deben sobrevivir reinicios. `langgraph-checkpoint-redis` exige RedisJSON y
RediSearch: vienen incluidos en Redis 8+ (la imagen oficial `redis:8.10.3` los compila y el
entrypoint los carga al arrancar `redis-server`, verificado en docker-library-redis).

## Decisión
- `AsyncRedisSaver.from_conn_string(REDIS_URL, ttl=...)` como async context manager en el
  lifespan; su `__aenter__` ya corre `asetup()`.
- El grafo recibe el checkpointer inyectado (lección de la pre-entrega 5).
- Redis con AOF (`appendonly yes`) y volumen nombrado: sobrevive también al reinicio de Redis.
- El estado del grafo usa solo tipos JSON (TypedDict) para no depender de la deserialización
  de clases propias.
- `InMemorySaver` solo en tests.

## Alternativas descartadas
- `AsyncPostgresSaver`: válido, pero suma un servicio; Redis ya se usa para las estadísticas BM25.

## Consecuencias
Probado contra Redis 8.10.3 real: un turno, nuevo saver + nuevo grafo (= reinicio), el
historial sigue y la conversación continúa.
