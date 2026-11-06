#!/usr/bin/env bash
# Aplica las mejoras del intelligence-system sobre tu carpeta del repo.
# Ejecutar DESDE ~/ai-engineering/intelligence_system:   bash mejoras/aplicar.sh
# - Copia cada archivo de mejoras/intelligence_system/ a su lugar, guardando el anterior como <archivo>.orig
# - Escribe el workflow de CI en ../.github/workflows/ (raíz del repo)
# - Corre make test y make lint al final
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
grep -q 'name = "intelligence-system"' pyproject.toml 2>/dev/null || { echo "✗ Ejecutá esto desde ~/ai-engineering/intelligence_system (no encuentro pyproject.toml)"; exit 1; }
[ -d .git ] && { echo "✗ Estás en la raíz del repo; entrá a intelligence_system/"; exit 1; }

changed=0
while IFS= read -r -d '' src; do
  rel="${src#"$here/intelligence_system/"}"
  dest="./$rel"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
    echo "= $rel (sin cambios)"; continue
  fi
  if [ -f "$dest" ]; then cp -p "$dest" "$dest.orig"; echo "~ $rel (anterior → $rel.orig)"; else echo "+ $rel (nuevo)"; fi
  cp "$src" "$dest"; changed=$((changed+1))
done < <(find "$here/intelligence_system" -type f -print0)

mkdir -p ../.github/workflows
if [ -f ../.github/workflows/intelligence-system.yml ] && cmp -s "$here/.github/workflows/intelligence-system.yml" ../.github/workflows/intelligence-system.yml; then
  echo "= .github/workflows/intelligence-system.yml (sin cambios)"
else
  cp "$here/.github/workflows/intelligence-system.yml" ../.github/workflows/intelligence-system.yml
  echo "+ ../.github/workflows/intelligence-system.yml"; changed=$((changed+1))
fi

echo; echo "Archivos aplicados: $changed. Verificando..."; echo
make test 2>&1 | tail -1
make lint 2>&1 | grep -E "All checks passed|already formatted|error|would reformat" || true
echo
echo "Para revisar un cambio:   diff README.md.orig README.md"
echo "Para borrar los respaldos: find . -name '*.orig' -delete"
