"""
Ejercicio: Orquestador concurrente de modelos de IA
Unidad: Asyncio en Ingeniería de IA (Python 3.12+)

Simula un backend que debe consultar múltiples LLMs de forma eficiente,
con control de concurrencia, timeouts y resiliencia ante fallos.
"""

import asyncio
import random
import time
import logging

# --- Configuración de logging para ver el orden temporal de ejecución ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# 1. SIMULACIÓN DE LLAMADAS A MODELOS
# ============================================================
# Cada corrutina simula una llamada a un LLM con latencia variable.
# Usamos await asyncio.sleep (NO time.sleep) para no bloquear el loop.

async def gpt_4_call(request_id: int) -> dict:
    latency = random.uniform(0.3, 1.5)
    log.info(f"[req={request_id}] GPT-4        → iniciando (latencia simulada: {latency:.2f}s)")
    await asyncio.sleep(latency)
    log.info(f"[req={request_id}] GPT-4        ✓ respondió")
    return {"model": "gpt-4", "request_id": request_id, "latency": latency}


async def claude_3_call(request_id: int) -> dict:
    latency = random.uniform(0.3, 1.5)
    log.info(f"[req={request_id}] Claude-3     → iniciando (latencia simulada: {latency:.2f}s)")
    await asyncio.sleep(latency)
    log.info(f"[req={request_id}] Claude-3     ✓ respondió")
    return {"model": "claude-3", "request_id": request_id, "latency": latency}


async def local_llama_call(request_id: int) -> dict:
    # Deliberadamente más lento para forzar timeouts ocasionales
    latency = random.uniform(0.5, 3.0)
    log.info(f"[req={request_id}] Llama-local  → iniciando (latencia simulada: {latency:.2f}s)")
    await asyncio.sleep(latency)
    log.info(f"[req={request_id}] Llama-local  ✓ respondió")
    return {"model": "llama-local", "request_id": request_id, "latency": latency}


# ============================================================
# 2. ORQUESTADOR DE UNA SOLA REQUEST (3 modelos en paralelo)
# ============================================================
# Dispara las 3 llamadas concurrentemente con gather + timeout global.
# return_exceptions=True hace que un fallo individual no tumbe el resto.

async def consultar_modelos(request_id: int, sem: asyncio.Semaphore) -> dict:
    """
    Procesa UNA consulta disparando los 3 modelos en paralelo.
    El semáforo limita cuántas de estas 'consultas' corren simultáneamente.
    """
    async with sem:
        log.info(f"[req={request_id}] ╔══ CONSULTA INICIADA (semáforo adquirido)")
        inicio = time.perf_counter()

        try:
            # Timeout de 2s para TODA la consulta (los 3 modelos juntos)
            async with asyncio.timeout(2.0):
                resultados = await asyncio.gather(
                    gpt_4_call(request_id),
                    claude_3_call(request_id),
                    local_llama_call(request_id),
                    return_exceptions=True,  # resiliencia ante fallos individuales
                )

            elapsed = time.perf_counter() - inicio
            exitos = [r for r in resultados if isinstance(r, dict)]
            errores = [r for r in resultados if isinstance(r, Exception)]

            log.info(
                f"[req={request_id}] ╚══ CONSULTA OK en {elapsed:.2f}s "
                f"({len(exitos)} éxitos, {len(errores)} errores)"
            )
            return {
                "request_id": request_id,
                "status": "ok",
                "elapsed": elapsed,
                "resultados": exitos,
                "errores": [str(e) for e in errores],
            }

        except TimeoutError:
            elapsed = time.perf_counter() - inicio
            log.warning(
                f"[req={request_id}] ╚══ TIMEOUT a los {elapsed:.2f}s "
                f"(la consulta superó los 2s permitidos)"
            )
            return {
                "request_id": request_id,
                "status": "timeout",
                "elapsed": elapsed,
            }


# ============================================================
# 3. PUNTO DE ENTRADA PRINCIPAL
# ============================================================

async def main():
    TOTAL_CONSULTAS = 10
    CONCURRENCIA_MAX = 2  # requisito del ejercicio: solo 2 simultáneas

    log.info("=" * 70)
    log.info(f"Lanzando {TOTAL_CONSULTAS} consultas | max concurrencia: {CONCURRENCIA_MAX}")
    log.info("=" * 70)

    sem = asyncio.Semaphore(CONCURRENCIA_MAX)
    inicio_total = time.perf_counter()

    # Encolamos las 10 consultas; el semáforo decide cuántas corren a la vez
    tareas = [consultar_modelos(i, sem) for i in range(1, TOTAL_CONSULTAS + 1)]
    resultados = await asyncio.gather(*tareas)

    elapsed_total = time.perf_counter() - inicio_total

    # --- Resumen final ---
    log.info("=" * 70)
    log.info("RESUMEN FINAL")
    log.info("=" * 70)

    ok = [r for r in resultados if r["status"] == "ok"]
    timeouts = [r for r in resultados if r["status"] == "timeout"]

    log.info(f"Total consultas:    {TOTAL_CONSULTAS}")
    log.info(f"Exitosas:           {len(ok)}")
    log.info(f"Timeouts:           {len(timeouts)}")
    log.info(f"Tiempo total:       {elapsed_total:.2f}s")
    log.info(
        f"Tiempo secuencial estimado (sin concurrencia): "
        f"~{TOTAL_CONSULTAS * 1.5:.1f}s"
    )
    log.info(f"Speedup aprox:      {(TOTAL_CONSULTAS * 1.5) / elapsed_total:.1f}x")


if __name__ == "__main__":
    # asyncio.run es el estándar moderno en Python 3.12+
    # gestiona creación/cierre del loop automáticamente
    asyncio.run(main())