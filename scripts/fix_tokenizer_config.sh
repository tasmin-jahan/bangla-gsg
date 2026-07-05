#!/usr/bin/env bash
# Fixes saved/tokenizer/tokenizer_config.json:
#   1. tokenizer_class -> PreTrainedTokenizerFast
#   2. model_max_length -> 2048
#   3. padding_side -> left (needed for fine-tuning/inference batching; inert during packing)
#   4. clean_up_tokenization_spaces -> false
#
# Usage: ./fix_tokenizer_config.sh [path-to-tokenizer_config.json]
# Defaults to saved/tokenizer/tokenizer_config.json if no argument given.


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_PATH="${1:-$PROJECT_ROOT/saved/tokenizer/tokenizer_config.json}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: file not found: $CONFIG_PATH"
  exit 1
fi

BACKUP_PATH="${CONFIG_PATH}.bak.$(date +%Y%m%d%H%M%S)"
echo "Backing up original to ${BACKUP_PATH}"
cp "$CONFIG_PATH" "${BACKUP_PATH}"


echo "1. tokenizer_class -> PreTrainedTokenizerFast"
sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' "$CONFIG_PATH"


echo "2. model_max_length -> 2048"
sed -i 's/"model_max_length": [0-9]*/"model_max_length": 2048/' "$CONFIG_PATH"


echo "3. padding_side -> left"
if grep -q '"padding_side"' "$CONFIG_PATH"; then
  echo "   (already present, skipping)"
else
  sed -i 's/"pad_token": "<pad>"/"pad_token": "<pad>",\n  "padding_side": "left"/' "$CONFIG_PATH"
fi


echo "4. clean_up_tokenization_spaces -> false"
if grep -q '"clean_up_tokenization_spaces"' "$CONFIG_PATH"; then
  echo "   (already present, skipping)"
else
  sed -i 's/"eos_token": "<\/s>"/"eos_token": "<\/s>",\n  "clean_up_tokenization_spaces": false/' "$CONFIG_PATH"
fi


echo ""
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

echo ""
echo "Validating JSON..."
"$PYTHON" -m json.tool "$CONFIG_PATH" > /dev/null && echo "  valid JSON"

echo ""
echo "Loading with transformers to confirm..."
"$PYTHON" - "$CONFIG_PATH" <<'EOF'
import sys, os
from transformers import AutoTokenizer
config_path = sys.argv[1]
tok_dir = os.path.dirname(config_path)
tok = AutoTokenizer.from_pretrained(tok_dir)
print("  class:", type(tok).__name__)
print("  max_length:", tok.model_max_length)
print("  padding_side:", tok.padding_side)
print("  clean_up_tokenization_spaces:", tok.clean_up_tokenization_spaces)
EOF

echo ""
echo "Done. Original file backed up at ${BACKUP_PATH}"
