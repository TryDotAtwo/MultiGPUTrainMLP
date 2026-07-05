#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${CUTLASS_ROOT:-}" != "" ] && [ -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]; then
  echo "cutlass_root=${CUTLASS_ROOT}"
  return 0 2>/dev/null || exit 0
fi

for candidate in "${MGT_CUTLASS_ROOT:-}" /opt/cutlass /usr/local/cutlass; do
  if [ "$candidate" != "" ] && [ -f "$candidate/include/cutlass/cutlass.h" ]; then
    export CUTLASS_ROOT="$candidate"
    echo "cutlass_root=${CUTLASS_ROOT}"
    return 0 2>/dev/null || exit 0
  fi
done

deps_dir="${MGT_DEPS_DIR:-${repo_root}/.deps}"
cutlass_root="${MGT_CUTLASS_ROOT:-${deps_dir}/cutlass}"
cutlass_ref="${MGT_CUTLASS_REF:-main}"

if [ ! -f "${cutlass_root}/include/cutlass/cutlass.h" ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to fetch CUTLASS; set CUTLASS_ROOT or MGT_CUTLASS_ROOT to an existing checkout" >&2
    exit 2
  fi
  mkdir -p "$deps_dir"
  git clone --depth 1 --branch "$cutlass_ref" https://github.com/NVIDIA/cutlass.git "$cutlass_root"
fi

export CUTLASS_ROOT="$cutlass_root"
echo "cutlass_root=${CUTLASS_ROOT}"