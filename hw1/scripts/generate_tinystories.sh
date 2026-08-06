#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$PROJECT_ROOT/main.py" generate \
  --checkpoint "$PROJECT_ROOT/pretrained_models/tinystories_lm.pt" \
  --device "${DEVICE:-cuda:6}" \
  --prompt "${PROMPT:-Once upon a time}" \
  "$@"
