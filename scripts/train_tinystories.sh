#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$PROJECT_ROOT/main.py" train \
  --data "$PROJECT_ROOT/pretrained_models/tinystories_tokens.npy" \
  --checkpoint "$PROJECT_ROOT/pretrained_models/tinystories_lm.pt" \
  --device "${DEVICE:-cuda:6}" \
  "$@"
