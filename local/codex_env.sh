#!/usr/bin/env bash
# Source this file to export AZURE_OPENAI_API_KEY for the Codex CLI.
# Usage: source local/codex_env.sh

set -euo pipefail

_DEFAULT_KEY_FILES=(
  "${AZURE_KEY_FILE:-}"
  "/Users/yiminglin/Documents/Codebase/api_keys/azure_cloudbank/gpt-54_1.txt"
  "$HOME/api_keys/azure_cloudbank/gpt-54_1.txt"
)

_KEY_FILE=""
for candidate in "${_DEFAULT_KEY_FILES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    _KEY_FILE="$candidate"
    break
  fi
done

if [[ -z "$_KEY_FILE" ]]; then
  echo "Azure key file not found. Set AZURE_KEY_FILE or place gpt-54_1.txt in a known location." >&2
  return 1 2>/dev/null || exit 1
fi

_API_KEY="$(
  awk -F': *' '$1 == "api_key" { print $2; exit }' "$_KEY_FILE"
)"
if [[ -z "$_API_KEY" ]]; then
  echo "Failed to parse api_key from $_KEY_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

export AZURE_OPENAI_API_KEY="$_API_KEY"
echo "AZURE_OPENAI_API_KEY exported (${#AZURE_OPENAI_API_KEY} chars) from $_KEY_FILE"
