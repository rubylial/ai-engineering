"""
Script de validación de la pre-entrega 1.

Carga .env, instancia el AsyncLLMManager según LLM_PROVIDER, y hace
la misma pregunta ('¿Qué es la entropía?') en dos modos:
  1. Modo normal — espera la respuesta completa.
  2. Modo streaming — imprime los tokens a medida que llegan.

Captura errores controlados: si falla la API key, rate limit o red,
imprime un mensaje legible en lugar de un traceback crudo.
"""

import asyncio
import sys
import time

from dotenv import load_dotenv

from llm_client import (
    AsyncLLMManager,
    ChatMessage,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    ModelConfig,
    Provider,
    Role,
)


PREGUNTA = "¿Qué es la entropía? Responde en 3 oraciones."


# Modelos por defecto por proveedor — podés cambiarlos vía argumento.
DEFAULT_MODEL = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
}


def _build_model_config(provider: Provider) -> ModelConfig:
    return ModelConfig(
        model=DEFAULT_MODEL[provider],
        temperature=0.3,       # respuesta más factual, menos creativa
        max_tokens=300,
        system_prompt="Eres un profesor de física que responde con claridad y brevedad.",
    )


async def probar_modo_normal(manager: AsyncLLMManager) -> None:
    print("\n" + "=" * 70)
    print(f"MODO NORMAL — {manager.provider.value}")
    print("=" * 70)

    messages = [ChatMessage(role=Role.USER, content=PREGUNTA)]
    config = _build_model_config(manager.provider)

    t0 = time.perf_counter()
    response = await manager.generate(messages, config)
    elapsed = time.perf_counter() - t0

    print(f"\nRespuesta ({elapsed:.2f}s):\n{response.content}")
    print(
        f"\nModelo: {response.model} | "
        f"Tokens in/out: {response.usage.input_tokens}/{response.usage.output_tokens} | "
        f"finish: {response.finish_reason}"
    )


async def probar_modo_streaming(manager: AsyncLLMManager) -> None:
    print("\n" + "=" * 70)
    print(f"MODO STREAMING — {manager.provider.value}")
    print("=" * 70)

    messages = [ChatMessage(role=Role.USER, content=PREGUNTA)]
    config = _build_model_config(manager.provider)

    t0 = time.perf_counter()
    ttft: float | None = None
    total_chars = 0

    print("\nRespuesta: ", end="", flush=True)
    async for token in manager.stream(messages, config):
        if ttft is None:
            # Time-To-First-Token: métrica clave de UX en productos LLM
            ttft = time.perf_counter() - t0
        print(token, end="", flush=True)
        total_chars += len(token)

    elapsed = time.perf_counter() - t0
    print(
        f"\n\nTTFT: {ttft:.2f}s | total: {elapsed:.2f}s | "
        f"{total_chars} chars"
    )


async def main() -> int:
    # 1. Cargar .env (NO rompe si no existe — útil cuando las vars
    #    vienen del entorno del contenedor/CI).
    load_dotenv()

    # 2. Construir manager — falla acá si falta la key del proveedor elegido.
    try:
        manager = AsyncLLMManager.from_env()
    except LLMConfigError as e:
        print(f"✗ Error de configuración: {e}")
        print("  Copia .env.example a .env y completa las variables.")
        return 2

    # 3. Ejecutar las dos pruebas, capturando errores de red/API de forma controlada.
    try:
        await probar_modo_normal(manager)
        await probar_modo_streaming(manager)
    except LLMAuthError as e:
        print(f"\n✗ Autenticación fallida: {e}")
        print("  Verifica tu API key.")
        return 3
    except LLMRateLimitError as e:
        print(f"\n✗ Rate limit alcanzado: {e}")
        print("  Esperá unos segundos o bajá el throughput.")
        return 4
    except LLMConnectionError as e:
        print(f"\n✗ Problema de red: {e}")
        return 5
    except LLMError as e:
        # Cualquier otro error del proveedor — no reventamos al usuario.
        print(f"\n✗ Error del proveedor: {e}")
        return 6

    print("\n✓ Pruebas completadas correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
