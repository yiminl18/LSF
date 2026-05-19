#!/usr/bin/env bash
# Source this file to export AZURE_OPENAI_API_KEY for the Codex CLI.
# Usage:  source local/codex_env.sh   (from the LSF project root)

_KEY_FILE="/Users/yiminglin/Documents/Codebase/api_keys/azure_cloudbank/gpt-54_1.txt"
export AZURE_OPENAI_API_KEY="$(grep -oP 'api_key: \K\S+' "$_KEY_FILE")"
echo "AZURE_OPENAI_API_KEY exported (${#AZURE_OPENAI_API_KEY} chars)"
