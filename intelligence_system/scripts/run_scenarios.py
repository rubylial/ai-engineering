"""Corre los escenarios de evidencia contra la API levantada y deja el registro en docs/traces/.

Uso (con el stack arriba):
    uv run python scripts/run_scenarios.py            # o: make scenarios
Variables: API_BASE_URL (http://localhost:8000), API_KEY, PHOENIX_UI_URL (http://localhost:6006)

Cada escenario imprime su trace_id y el link directo a la traza en Phoenix
(`/redirects/traces/<trace_id>`). Al final reescribe la tabla de
`docs/traces/traces.md` y guarda el detalle en `docs/traces/scenario_results.json`.
COSTO con Anthropic: ~25 llamadas a claude-haiku-5-5 en total (centavos de dólar).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
MD_START, MD_END = "<!-- SCENARIOS:START -->", "<!-- SCENARIOS:END -->"


@dataclass
class Outcome:
    id: str
    name: str
    input: str
    expected: str
    obtained: str = ""
    passed: bool = False
    status_code: int | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    request_id: str | None = None
    thread_id: str | None = None
    latency_ms: float | None = None  # del request principal del escenario, medido por el cliente
    details: dict[str, Any] = field(default_factory=dict)


def _nodes(body: dict[str, Any]) -> list[str]:
    return [s["node"] for s in body.get("steps", [])]


class Runner:
    def __init__(self, client: httpx.AsyncClient, phoenix_url: str) -> None:
        self.client = client
        self.phoenix_url = phoenix_url.rstrip("/")
        self.suffix = uuid.uuid4().hex[:6]
        self.last_latency_ms: float | None = None

    def trace_url(self, trace_id: str | None) -> str | None:
        return f"{self.phoenix_url}/redirects/traces/{trace_id}" if trace_id else None

    async def chat(self, message: str, thread_id: str, **extra: Any) -> httpx.Response:
        started = time.perf_counter()
        resp = await self.client.post("/v1/chat", json={"message": message, "thread_id": thread_id, **extra})
        self.last_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return resp

    def _fill(self, outcome: Outcome, resp: httpx.Response) -> dict[str, Any]:
        body = resp.json()
        outcome.status_code = resp.status_code
        outcome.trace_id = body.get("trace_id")
        outcome.trace_url = self.trace_url(outcome.trace_id)
        outcome.request_id = body.get("request_id") or resp.headers.get("X-Request-ID")
        outcome.thread_id = body.get("thread_id", outcome.thread_id)
        outcome.latency_ms = self.last_latency_ms
        return body

    async def rag_simple(self) -> Outcome:
        q = "¿Para qué sirve asyncio.Semaphore cuando llamo a un LLM?"
        o = Outcome("1", "Pregunta RAG simple", q, "grounded=true, ≥1 cita de documento, pasa por researcher")
        o.thread_id = f"esc1-{self.suffix}"
        body = self._fill(o, await self.chat(q, o.thread_id))
        cites = body.get("citations", [])
        o.passed = o.status_code == 200 and body.get("grounded") is True and bool(cites)
        o.passed = o.passed and "researcher" in _nodes(body)
        o.obtained = f"grounded={body.get('grounded')}, citas={len(cites)}, pasos={'→'.join(_nodes(body))}"
        o.details = {"answer": body.get("answer"), "citations": [c["id"] for c in cites]}
        return o

    async def multi_step(self) -> Outcome:
        q = (
            "Según la documentación, ¿con qué tamaño de chunk y overlap conviene empezar? "
            "Calculá cuántos chunks salen de un documento de 12000 tokens con esa configuración."
        )
        o = Outcome(
            "2", "Tarea multi-paso (≥2 workers)", q, "researcher + analyst, citas de documento y de cálculo"
        )
        o.thread_id = f"esc2-{self.suffix}"
        body = self._fill(o, await self.chat(q, o.thread_id))
        nodes = _nodes(body)
        kinds = {c["kind"] for c in body.get("citations", [])}
        o.passed = (
            o.status_code == 200 and {"researcher", "analyst"} <= set(nodes) and body.get("grounded") is True
        )
        o.obtained = f"pasos={'→'.join(nodes)}, tipos de cita={sorted(kinds)}"
        o.details = {"answer": body.get("answer")}
        return o

    async def no_evidence(self) -> Outcome:
        q = "¿Cuál es la capital de Australia?"
        o = Outcome("3", "Pregunta sin evidencia", q, "grounded=false, 0 citas, responde que no sabe")
        o.thread_id = f"esc3-{self.suffix}"
        body = self._fill(o, await self.chat(q, o.thread_id))
        o.passed = o.status_code == 200 and body.get("grounded") is False and not body.get("citations")
        o.obtained = f"grounded={body.get('grounded')}, citas={len(body.get('citations', []))}"
        o.details = {"answer": body.get("answer")}
        return o

    async def continuation(self) -> Outcome:
        first, second = "¿Qué es BM25?", "¿Y qué limitación tiene frente a la búsqueda semántica?"
        o = Outcome(
            "4",
            "Continuación por thread_id",
            f"{first} → {second}",
            "2º turno resuelto con el contexto; 4 mensajes",
        )
        o.thread_id = f"esc4-{self.suffix}"
        await self.chat(first, o.thread_id)
        body = self._fill(o, await self.chat(second, o.thread_id))
        history = (await self.client.get(f"/v1/threads/{o.thread_id}")).json()
        roles = [m["role"] for m in history.get("messages", [])]
        o.passed = o.status_code == 200 and roles == ["user", "assistant", "user", "assistant"]
        o.passed = o.passed and body.get("grounded") is True
        o.obtained = f"mensajes en el thread={len(roles)}, grounded 2º turno={body.get('grounded')}"
        o.details = {
            "answer": body.get("answer"),
            "supervisor_task": body.get("steps", [{}])[0].get("detail"),
        }
        return o

    async def controlled_error(self) -> Outcome:
        q = "Explicá RRF"
        o = Outcome(
            "5",
            "Error controlado / timeout",
            f"{q} (deadline_s=0.001) + mensaje vacío",
            "504 problem+json con request_id y trace_id; 422 para el mensaje vacío",
        )
        o.thread_id = f"esc5-{self.suffix}"
        resp = await self.chat(q, o.thread_id, deadline_s=0.001)
        body = self._fill(o, resp)
        invalid = await self.client.post("/v1/chat", json={"message": ""})
        o.passed = resp.status_code == 504 and bool(body.get("request_id")) and invalid.status_code == 422
        o.obtained = f"{resp.status_code} {body.get('type')}; mensaje vacío → {invalid.status_code}"
        o.details = {"timeout_body": body, "validation_body": invalid.json()}
        return o

    async def streaming(self) -> Outcome:
        q = "¿Qué es Reciprocal Rank Fusion?"
        o = Outcome("6", "Streaming SSE (extra)", q, "eventos step → token → citation → done")
        o.thread_id = f"esc6-{self.suffix}"
        kinds: list[str] = []
        done: dict[str, Any] = {}
        started = time.perf_counter()
        async with self.client.stream(
            "POST", "/v1/chat/stream", json={"message": q, "thread_id": o.thread_id}
        ) as resp:
            o.status_code = resp.status_code
            event = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    kinds.append(event)
                    if event in ("done", "error"):
                        done = json.loads(line.split(":", 1)[1])
        o.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        o.trace_id = done.get("trace_id")
        o.trace_url = self.trace_url(o.trace_id)
        o.request_id = done.get("request_id")
        o.passed = bool(kinds) and kinds[-1] == "done" and "step" in kinds
        o.obtained = (
            f"eventos={len(kinds)} (step={kinds.count('step')}, token={kinds.count('token')}, "
            f"citation={kinds.count('citation')}), último={kinds[-1] if kinds else '-'}"
        )
        return o


def write_markdown(outcomes: list[Outcome], path: Path) -> None:
    rows = [
        "| # | Escenario | Input | Esperado | Obtenido | OK | Latencia | trace_id |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        trace = f"[`{o.trace_id[:12]}…`]({o.trace_url})" if o.trace_id and o.trace_url else "—"
        latency = f"{o.latency_ms / 1000:.1f} s" if o.latency_ms is not None else "—"
        rows.append(
            f"| {o.id} | {o.name} | {o.input} | {o.expected} | {o.obtained} | "
            f"{'✅' if o.passed else '❌'} | {latency} | {trace} |"
        )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    block = f"{MD_START}\n_Última corrida: {stamp}_\n\n" + "\n".join(rows) + f"\n{MD_END}"
    text = path.read_text(encoding="utf-8") if path.exists() else f"{MD_START}\n{MD_END}\n"
    start, end = text.find(MD_START), text.find(MD_END)
    if start == -1 or end == -1:
        text = f"{text.rstrip()}\n\n{block}\n"
    else:
        text = text[:start] + block + text[end + len(MD_END) :]
    path.write_text(text, encoding="utf-8")


async def run_all(base_url: str, api_key: str, phoenix_url: str) -> list[Outcome]:
    headers = {"X-API-Key": api_key}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=180) as client:
        ready = await client.get("/ready")
        if ready.status_code != 200:
            print(f"⚠ /ready respondió {ready.status_code}: {ready.text}", file=sys.stderr)
        runner = Runner(client, phoenix_url)
        outcomes: list[Outcome] = []
        for scenario in (
            runner.rag_simple,
            runner.multi_step,
            runner.no_evidence,
            runner.continuation,
            runner.controlled_error,
            runner.streaming,
        ):
            outcome = await scenario()
            outcomes.append(outcome)
            mark = "✅" if outcome.passed else "❌"
            latency = f" · {outcome.latency_ms / 1000:.1f}s" if outcome.latency_ms is not None else ""
            print(f"{mark} [{outcome.id}] {outcome.name}: {outcome.obtained}{latency}")
            print(f"    trace_id={outcome.trace_id}  {outcome.trace_url or ''}")
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--phoenix-url", default=os.getenv("PHOENIX_UI_URL", "http://localhost:6006"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "traces"))
    args = parser.parse_args()
    if not args.api_key:
        print("Falta API_KEY (variable de entorno o --api-key).", file=sys.stderr)
        return 2

    outcomes = asyncio.run(run_all(args.base_url, args.api_key, args.phoenix_url))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario_results.json").write_text(
        json.dumps([asdict(o) for o in outcomes], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(outcomes, out_dir / "traces.md")
    passed = sum(o.passed for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} escenarios OK · detalle en {out_dir / 'scenario_results.json'}")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
