#!/usr/bin/env python3
"""check_secrets.py — auditoría exhaustiva de secretos antes de commitear/pushear.

Revisa, SIN dependencias externas (solo `git` + stdlib), que no se filtren claves ni
valores sensibles. Cubre las tres superficies donde un secreto puede escaparse:

  1. Árbol trackeado actual   (git ls-files)            -> lo versionado hoy
  2. Cambios staged           (git diff --cached)        -> lo que estás por commitear
  3. Commits sin pushear      (origin/<branch>..HEAD)    -> lo que saldría en el push
  4. Archivos sensibles        (.env, *.pem, *.key, ...) -> que NO estén trackeados
  5. .gitignore                (.env ignorado)

Un secreto en la historia (3) igual bloquea el push y se filtra aunque el árbol (1)
esté limpio: por eso se escanea el rango que se va a pushear, no solo los archivos.

Detecta claves de proveedor conocidas (Anthropic, OpenAI, Pinecone, AWS, GCP, GitHub,
Slack), bloques de clave privada y JWT (severidad FAIL), y como heurística
asignaciones tipo API_KEY=/SECRET=/TOKEN= con un valor que no es un placeholder, más
hex/UUID de alta entropía (severidad WARN).

Severidad:
  FAIL -> casi seguro un secreto real; corta con exit 1.
  WARN -> sospechoso, revisalo a mano; con --strict también corta.

Los valores se imprimen SIEMPRE enmascarados (nunca el secreto completo).

Uso:
  python3 scripts/check_secrets.py                 # repo entero, rango sin pushear
  python3 scripts/check_secrets.py --path intelligence_system
  python3 scripts/check_secrets.py --staged        # solo lo staged (hook pre-commit)
  python3 scripts/check_secrets.py --all-history   # toda la historia, no solo sin pushear
  python3 scripts/check_secrets.py --strict        # WARN también corta

Ignorar un falso positivo puntual: agregá  # pragma: allowlist secret  al final de la línea.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SELF = Path(__file__).name
ALLOWLIST_MARK = "pragma: allowlist secret"

# Directorios/archivos que no aportan secretos y sí mucho ruido (hashes, trace_ids, locks).
SKIP_DIR_PARTS = (
    ".git/", ".venv/", "venv/", "env/", "node_modules/", "__pycache__/",
    ".mypy_cache/", ".ruff_cache/", ".pytest_cache/", "dist/", "build/", ".idea/",
    "site-packages/",
)
# En estos los chequeos WARN (hex/UUID/entropía) se silencian; los FAIL igual corren.
NOISE_PATH_PARTS = ("docs/traces/",)
NOISE_SUFFIXES = (".lock", ".lockb", ".min.js", ".map", ".ipynb")
LOCK_NAMES = ("uv.lock", "package-lock.json", "poetry.lock", "pnpm-lock.yaml", "yarn.lock")
# Archivos de configuración donde un hex/UUID suelto puede ser un secreto (no un hash/ejemplo).
CONFIG_SUFFIXES = (".env", ".ini", ".cfg", ".conf", ".properties", ".toml", ".yaml", ".yml")

# Nombres de archivo que NO deberían estar jamás versionados.
SENSITIVE_FILE_RES = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example$|sample$|template$|dist$)[^/]+$"),
    re.compile(r"\.pem$"), re.compile(r"\.key$"), re.compile(r"\.pfx$"),
    re.compile(r"\.p12$"), re.compile(r"\.keystore$"), re.compile(r"\.jks$"),
    re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"),
    re.compile(r"(^|/)\.npmrc$"), re.compile(r"(^|/)\.pypirc$"),
    re.compile(r"(^|/)\.htpasswd$"),
    re.compile(r"(^|/)credentials$"), re.compile(r"(^|/)\.netrc$"),
    re.compile(r"service[-_]?account.*\.json$"),
    re.compile(r"(^|/)\.aws/credentials$"),
]

FAIL = "FAIL"
WARN = "WARN"

# (nombre, severidad, regex). Los FAIL son patrones de proveedor muy específicos.
PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("Anthropic API key", FAIL, re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI project key", FAIL, re.compile(r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}")),
    ("OpenAI classic key", FAIL, re.compile(r"sk-[A-Za-z0-9]{32,}")),
    ("Pinecone key", FAIL, re.compile(r"pcsk_[A-Za-z0-9_]{20,}")),
    ("AWS access key id", FAIL, re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[A-Z0-9]{16}\b")),
    ("Google API key", FAIL, re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("GitHub token", FAIL, re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("Slack token", FAIL, re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Clave privada (bloque PEM)", FAIL, re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("JWT", FAIL, re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
]

# Patrones ruidosos (hex/UUID): SOLO se corren en archivos de configuración tipo
# .env/.ini/.toml/.yaml, donde un valor así suele ser un secreto. En .py/.md/.json
# son hashes, ejemplos o trace_ids legítimos, así que ahí no se miran.
CONFIG_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("Hex largo (≥32, posible secreto)", WARN, re.compile(r"\b[0-9a-f]{32,}\b")),
    ("UUID (posible clave Pinecone legacy)", WARN, re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")),
]

# Asignaciones sospechosas: NOMBRE_SENSIBLE = valor
ASSIGN_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWD|PASSWORD|PRIVATE[_-]?KEY|"
    r"ACCESS[_-]?KEY|CLIENT[_-]?SECRET|AUTH[_-]?TOKEN))\b\s*[:=]\s*(.+)"
)
# Un hallazgo de asignación solo cuenta si el VALOR parece un secreto de verdad: un token
# contiguo (sin espacios) de ≥16 caracteres. Esto descarta tipos (`str`, `SecretStr | None`),
# referencias (`$API_KEY`, `os.getenv(...)`) y ejemplos de comandos en la doc.
SECRETISH_RE = re.compile(r"^[A-Za-z0-9+/_=.\-]{16,}$")
# Un valor es placeholder (no secreto) si matchea esto tras quitar comillas/espacios.
PLACEHOLDER_RE = re.compile(
    r"""^(?:
        | x+ | \.{2,} | -+ | _+ | \*+ | 0+ | null | nil | none | false | true
        | (?:your|my|the|a|an|some)[-_ ]?.*
        | change[-_ ]?(?:me|this|it)?.* | cambia.* | replace.* | fill.* | set[-_ ]?.*
        | placeholder.* | example.* | ejemplo.* | sample.* | dummy.* | fake.*
        | mock.* | test.* | todo | tbd | n/?a | redacted.* | xxx.*
        | \$\{.*\} | \{\{.*\}\} | <.*> | %.*% | \$\(.*\)
        | os\.environ.* | process\.env.* | config\..* | settings\..*
        | getenv.* | secretmanager.* | arn:aws:secretsmanager:.* | vault:.*
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class Finding:
    scope: str          # "árbol", "staged", "historia"
    where: str          # path o commit:path
    line: int
    rule: str
    severity: str
    masked: str


def git(args: list[str], root: Path, check: bool = True) -> str | None:
    """Corre un comando git; devuelve stdout (str) o None si falló (con check=False)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=check,
        )
    except subprocess.CalledProcessError:
        return None
    return out.stdout


def mask(value: str) -> str:
    v = value.strip().strip("'\"").strip()
    if len(v) <= 8:
        return "*" * max(len(v), 1)
    return f"{v[:4]}…{v[-4:]} (len={len(v)})"


def shannon(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def in_skip_dir(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(part in p for part in SKIP_DIR_PARTS)


def is_noise(path: str) -> bool:
    """Archivos de datos donde hex/UUID/entropía son esperables (no secretos)."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    return (
        any(part in p for part in NOISE_PATH_PARTS)
        or any(p.endswith(suf) for suf in NOISE_SUFFIXES)
        or name in LOCK_NAMES
    )


def is_config(path: str) -> bool:
    """Archivo de configuración donde un hex/UUID suelto sí puede ser un secreto."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    return name.startswith(".env") or any(p.endswith(s) for s in CONFIG_SUFFIXES)


def scan_line(path: str, lineno: int, text: str, scope: str) -> list[Finding]:
    """Aplica todos los detectores a una línea. Devuelve findings (0..n)."""
    if ALLOWLIST_MARK in text:
        return []
    found: list[Finding] = []
    noise = is_noise(path)
    is_self = path.replace("\\", "/").rsplit("/", 1)[-1] == SELF

    # 1) Patrones de proveedor (FAIL): se miran en TODO archivo, incluso datos/ruido:
    #    una clave real nunca debería estar ahí.
    for name, severity, rx in PATTERNS:
        m = rx.search(text)
        if m:
            found.append(Finding(scope, path, lineno, name, severity, mask(m.group(0))))

    # 2) Patrones ruidosos (hex/UUID, WARN): solo en archivos de config y nunca en ruido.
    if is_config(path) and not noise:
        for name, severity, rx in CONFIG_PATTERNS:
            m = rx.search(text)
            if m:
                found.append(Finding(scope, path, lineno, name, severity, mask(m.group(0))))

    # 3) Heurística de asignación NOMBRE_SENSIBLE = valor. Solo cuenta si el valor parece
    #    un secreto real: token contiguo ≥16 y entropía ≥3 (descarta tipos y referencias).
    if not is_self and not noise:
        am = ASSIGN_RE.search(text)
        if am:
            raw = am.group(2).strip()
            raw = re.split(r"\s+#", raw, maxsplit=1)[0].strip()  # corta comentario inline
            val = raw.strip("'\"").strip()
            already = any(f.line == lineno and f.severity == FAIL for f in found)
            if (
                val
                and not already
                and not PLACEHOLDER_RE.match(val)
                and SECRETISH_RE.match(val)
                and shannon(val) >= 3.0
            ):
                found.append(Finding(
                    scope, path, lineno, f"Asignación sensible ({am.group(1)})",
                    WARN, mask(val),
                ))
    return found


def scan_tree(root: Path, subpath: str | None) -> list[Finding]:
    listing = git(["ls-files", "-z", *( [subpath] if subpath else [] )], root) or ""
    files = [f for f in listing.split("\0") if f]
    findings: list[Finding] = []
    for rel in files:
        if in_skip_dir(rel):
            continue
        fp = root / rel
        try:
            data = fp.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8000]:  # binario
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(rel, i, line, "árbol"))
    return findings


def scan_diff(root: Path, staged: bool, subpath: str | None) -> list[Finding]:
    args = ["diff", "--cached" if staged else "diff", "-U0", "--no-color"]
    if subpath:
        args += ["--", subpath]
    out = git(args, root) or ""
    findings: list[Finding] = []
    cur_file = "?"
    lineno = 0
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:]
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if not in_skip_dir(cur_file):
                findings.extend(scan_line(cur_file, lineno, line[1:], "staged"))
            lineno += 1
    return findings


def resolve_range(root: Path, all_history: bool) -> tuple[str | None, str]:
    """Devuelve (rango_para_git_log, descripción). rango=None => saltar historia."""
    if all_history:
        return "", "toda la historia (HEAD)"
    up = git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], root, check=False)
    if up:
        up = up.strip()
        return f"{up}..HEAD", f"commits sin pushear ({up}..HEAD)"
    branch = (git(["rev-parse", "--abbrev-ref", "HEAD"], root, check=False) or "HEAD").strip()
    for ref in (f"origin/{branch}", "origin/main", "origin/master"):
        if git(["rev-parse", "--verify", "--quiet", ref], root, check=False) is not None:
            return f"{ref}..HEAD", f"commits sin pushear ({ref}..HEAD)"
    return None, "sin upstream detectable"


def scan_history(root: Path, rng: str, subpath: str | None) -> list[Finding]:
    args = ["log", "-p", "-U0", "--no-color", "--no-merges"]
    if rng:
        args.append(rng)
    if subpath:
        args += ["--", subpath]
    out = git(args, root) or ""
    findings: list[Finding] = []
    commit = "?"
    cur_file = "?"
    lineno = 0
    for line in out.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1][:10]
            continue
        if line.startswith("+++ b/"):
            cur_file = line[6:]
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if not in_skip_dir(cur_file):
                findings.extend(scan_line(f"{commit}:{cur_file}", lineno, line[1:], "historia"))
            lineno += 1
    return findings


def check_sensitive_tracked(root: Path, subpath: str | None) -> list[Finding]:
    listing = git(["ls-files", *( [subpath] if subpath else [] )], root) or ""
    out: list[Finding] = []
    for rel in listing.splitlines():
        if any(rx.search(rel) for rx in SENSITIVE_FILE_RES):
            out.append(Finding("archivo", rel, 0, "Archivo sensible TRACKEADO", FAIL, "—"))
    return out


def check_gitignore(root: Path) -> list[Finding]:
    # .env debería estar ignorado. git check-ignore sale 0 si el path está ignorado.
    r = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=root)
    if r.returncode != 0:
        return [Finding("gitignore", ".env", 0, ".env NO está en .gitignore", WARN, "—")]
    return []


# ---- salida -----------------------------------------------------------------

def c(code: str, s: str, on: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if on else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="limitar el escaneo a un subdirectorio (p. ej. intelligence_system)")
    ap.add_argument("--staged", action="store_true", help="solo lo staged (rápido, para hook pre-commit)")
    ap.add_argument("--all-history", action="store_true", help="escanear toda la historia, no solo lo sin pushear")
    ap.add_argument("--strict", action="store_true", help="tratar WARN como fallo (exit 1)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    top = git(["rev-parse", "--show-toplevel"], Path.cwd(), check=False)
    if not top:
        print("No estás dentro de un repo git.", file=sys.stderr)
        return 2
    root = Path(top.strip())
    color = sys.stdout.isatty() and not args.no_color

    findings: list[Finding] = []
    scopes_run: list[str] = []

    if args.staged:
        findings += scan_diff(root, staged=True, subpath=args.path)
        scopes_run.append("staged")
    else:
        findings += check_sensitive_tracked(root, args.path)
        findings += check_gitignore(root)
        findings += scan_tree(root, args.path)
        scopes_run.append("árbol trackeado")
        findings += scan_diff(root, staged=True, subpath=args.path)
        scopes_run.append("staged")
        rng, desc = resolve_range(root, args.all_history)
        if rng is None:
            findings.append(Finding(
                "historia", "-", 0,
                "No pude determinar el rango sin pushear (sin upstream). "
                "Corré con --all-history o configurá el upstream.",
                WARN, "—"))
        else:
            findings += scan_history(root, rng, args.path)
        scopes_run.append(desc)

    # dedupe (misma regla, mismo lugar, misma línea, mismo valor)
    seen = set()
    uniq: list[Finding] = []
    for f in findings:
        k = (f.scope, f.where, f.line, f.rule, f.masked)
        if k not in seen:
            seen.add(k)
            uniq.append(f)

    fails = [f for f in uniq if f.severity == FAIL]
    warns = [f for f in uniq if f.severity == WARN]

    print(c("1", f"Scanner de secretos — repo: {root}", color))
    print(f"  Superficies: {', '.join(scopes_run)}")
    print(f"  Resultado: {len(fails)} FAIL · {len(warns)} WARN\n")

    def show(group: list[Finding], label: str, code: str) -> None:
        if not group:
            return
        print(c(code, f"── {label} ({len(group)}) ──", color))
        for f in sorted(group, key=lambda x: (x.scope, x.where, x.line)):
            loc = f"{f.where}:{f.line}" if f.line else f.where
            print(f"  [{f.scope}] {loc}")
            print(f"      {f.rule} → {f.masked}")
        print()

    show(fails, "FAIL — secretos casi seguros", "31")
    show(warns, "WARN — revisar a mano", "33")

    if not fails and not warns:
        print(c("32", "✓ Sin hallazgos. No se detectaron secretos.", color))
    elif not fails:
        print(c("32", "✓ Sin FAIL. Revisá los WARN por las dudas.", color))
    else:
        print(c("31", "✗ Hay secretos probables. NO pushees hasta limpiarlos.", color))
        print("  Recordá: limpiar el archivo NO basta si ya está en un commit; hay que")
        print("  reescribir ese commit (git reset --soft / rebase) para sacarlo de la historia.")

    hard = len(fails) + (len(warns) if args.strict else 0)
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())