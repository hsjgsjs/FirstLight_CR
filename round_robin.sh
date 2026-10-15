#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
source "$root/native_runner/local_config.sh"
exec "${CR_AI_PYTHON:-python}" -m native_runner.round_robin "$@"
