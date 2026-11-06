"""Captura automática de la evidencia de Phoenix (screenshots de las trazas).

Lee los trace_id reales de docs/traces/scenario_results.json (lo genera `make scenarios`),
navega a cada traza en la UI de Phoenix, expande el árbol de spans y guarda un PNG por
escenario en docs/traces/. También captura la lista del proyecto.

NO es parte del runtime del sistema: es una utilidad de evidencia. Usa Playwright, que no
está en las dependencias del proyecto; se corre con `uv run --with playwright` (ver abajo).

Requisitos (una sola vez):
    uv run --with playwright playwright install chromium

Uso (con el stack arriba y make scenarios ya corrido):
    uv run --with playwright python scripts/capture_phoenix.py
    # opciones: --phoenix-url http://localhost:6006  --project intelligence-system
    #           --results docs/traces/scenario_results.json  --out-dir docs/traces
    #           --headed   (ver el navegador en vez de headless)
    #           --wait 5   (segundos de espera por página; subilo si tu máquina es lenta)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# id del escenario (en scenario_results.json) -> nombre de archivo de la captura.
SCENARIO_SHOTS = {
    "1": "02-esc1-rag-simple.png",
    "2": "04-esc2-multipaso.png",
    "3": "07-esc3-sin-evidencia.png",
    "4": "08-esc4-continuacion.png",
    "5": "09-esc5-error.png",
    "6": "10-esc6-stream.png",
}


def expand_span_tree(page: Any, max_clicks: int = 40) -> int:
    """Best-effort: abre todos los nodos colapsados del árbol de spans (aria-expanded=false)."""
    opened = 0
    for _ in range(max_clicks):
        collapsed = page.query_selector_all('[aria-expanded="false"]')
        if not collapsed:
            break
        progressed = False
        for node in collapsed:
            with suppress(Exception):
                if node.is_visible():
                    node.click(timeout=1000)
                    opened += 1
                    progressed = True
                    page.wait_for_timeout(150)
        if not progressed:
            break
    return opened


def maybe_login(page: Any, base: str, email: str, password: str, wait_s: float) -> str:
    """Si Phoenix pide login, lo completa. Devuelve 'login-ok', 'sin-login' o 'login-falló'."""
    with suppress(Exception):
        page.goto(base, wait_until="networkidle", timeout=45000)
    page.wait_for_timeout(int(wait_s * 1000))
    pw = page.query_selector('input[type="password"]')
    if not pw or not pw.is_visible():
        return "sin-login"  # la UI ya está abierta, no hay formulario de contraseña
    if not email or not password:
        return "login-falló"  # hay form pero no nos pasaron credenciales
    for sel in ('input[type="email"]', 'input[name="email"]', 'input[name="username"]', 'input[type="text"]'):
        el = page.query_selector(sel)
        if el and el.is_visible():
            with suppress(Exception):
                el.fill(email)
            break
    with suppress(Exception):
        pw.fill(password)
    # Submit: botón explícito o Enter en el campo de contraseña.
    clicked = False
    for name in ("Log In", "Login", "Sign In", "Iniciar", "Ingresar"):
        with suppress(Exception):
            btn = page.get_by_role("button", name=name, exact=False)
            if btn.count() > 0:
                btn.first.click(timeout=3000)
                clicked = True
                break
    if not clicked:
        with suppress(Exception):
            pw.press("Enter")
    page.wait_for_timeout(int(wait_s * 1000))
    # Éxito si ya no hay campo de contraseña visible.
    still = page.query_selector('input[type="password"]')
    return "login-ok" if not (still and still.is_visible()) else "login-falló"


def shoot(page: Any, url: str, dest: Path, *, wait_s: float, expand: bool) -> bool:
    try:
        page.goto(url, wait_until="networkidle", timeout=45000)
    except Exception as exc:  # networkidle puede no dispararse en SPAs; seguimos igual
        print(f"  · aviso navegando a {url}: {type(exc).__name__} (sigo y capturo lo que haya)")
    page.wait_for_timeout(int(wait_s * 1000))
    if expand:
        n = expand_span_tree(page)
        if n:
            print(f"  · {n} nodos expandidos")
        page.wait_for_timeout(600)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(dest), full_page=True)
        kb = dest.stat().st_size // 1024
        print(f"  ✓ {dest.name} ({kb} KB)")
        return True
    except Exception as exc:
        print(f"  ✗ no pude capturar {dest.name}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phoenix-url", default="http://localhost:6006")
    parser.add_argument("--project", default="intelligence-system")
    parser.add_argument("--results", type=Path, default=ROOT / "docs" / "traces" / "scenario_results.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "traces")
    parser.add_argument("--wait", type=float, default=5.0, help="segundos de espera por página")
    parser.add_argument("--headed", action="store_true", help="mostrar el navegador")
    parser.add_argument(
        "--email",
        default=os.getenv("PHOENIX_EMAIL", ""),
        help="email de login de Phoenix (o variable PHOENIX_EMAIL); si Phoenix no pide login, se ignora",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("PHOENIX_PASSWORD", ""),
        help="contraseña de login de Phoenix (o variable PHOENIX_PASSWORD)",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        print(
            "Falta Playwright. Corré:\n"
            "  uv run --with playwright playwright install chromium\n"
            "  uv run --with playwright python scripts/capture_phoenix.py",
            file=sys.stderr,
        )
        return 2

    if not args.results.exists():
        print(f"No encuentro {args.results}. Corré `make scenarios` primero.", file=sys.stderr)
        return 2
    scenarios = json.loads(args.results.read_text(encoding="utf-8"))
    base = args.phoenix_url.rstrip("/")
    ok = 0
    total = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1680, "height": 1050}, device_scale_factor=2)

        # 0) Login si Phoenix lo pide (version-21 trae auth activada por defecto).
        state = maybe_login(page, base, args.email, args.password, args.wait)
        if state == "login-ok":
            print("Login en Phoenix: OK")
        elif state == "sin-login":
            print("Phoenix no pidió login")
        else:
            print(
                "⚠ Phoenix pide login y no pude autenticarme. Pasá las credenciales:\n"
                "  uv run --with playwright python scripts/capture_phoenix.py "
                '--email "TU_EMAIL" --password "TU_PASS"\n'
                "  (o exportá PHOENIX_EMAIL y PHOENIX_PASSWORD). Sigo igual, pero las capturas "
                "van a mostrar la pantalla de login.",
                file=sys.stderr,
            )

        # 1) Lista de trazas del proyecto (home de Phoenix: muestra los proyectos y sus conteos).
        total += 1
        print("Capturando la lista del proyecto…")
        if shoot(page, base, args.out_dir / "01-lista-de-trazas.png", wait_s=args.wait, expand=False):
            ok += 1
        # Intento entrar al proyecto por su nombre para una lista más específica (best-effort).
        try:
            link = page.get_by_text(args.project, exact=False).first
            link.click(timeout=4000)
            page.wait_for_timeout(int(args.wait * 1000))
            page.screenshot(path=str(args.out_dir / "01b-proyecto-trazas.png"), full_page=True)
            print("  ✓ 01b-proyecto-trazas.png (lista dentro del proyecto)")
        except Exception:
            print("  · no pude entrar al proyecto por click; la 01 ya sirve de lista general")

        # 2) Una captura por escenario, navegando por trace_id.
        for sc in scenarios:
            name = SCENARIO_SHOTS.get(str(sc.get("id")))
            trace_id = sc.get("trace_id")
            if not name or not trace_id:
                continue
            total += 1
            print(f"Escenario {sc['id']} ({sc['name']}) → {name}")
            url = f"{base}/redirects/traces/{trace_id}"
            # El multipaso y los que tienen workers se benefician de expandir el árbol.
            expand = sc["id"] in {"1", "2", "4", "6"}
            if shoot(page, url, args.out_dir / name, wait_s=args.wait, expand=expand):
                ok += 1

        browser.close()

    print(f"\n{ok}/{total} capturas guardadas en {args.out_dir}")
    print("Revisalas; si alguna salió vacía o con el árbol colapsado, reintentá con --wait 8")
    print("o tomá esa a mano (sobre todo 04-esc2-multipaso.png, que es la más importante).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())