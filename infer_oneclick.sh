#!/usr/bin/env bash
# Public entrypoint: prepare device nodes before starting inference with F5.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/prepare_ar3_inference.sh" "$@"
