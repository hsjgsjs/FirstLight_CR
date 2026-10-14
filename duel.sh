#!/usr/bin/env bash
# One entry point for the CPU opening comparison.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
source "$root/native_runner/local_config.sh"
python=${CR_AI_PYTHON:-python}
if ! command -v "$python" >/dev/null 2>&1; then
  echo "找不到 Python：$python。请激活项目的 Python 环境，或在 .env 中设置 CR_AI_PYTHON。" >&2
  exit 1
fi
exec "$python" -m native_runner.duel "$@"
