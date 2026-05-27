# Codex CLI Setup (Azure gpt-5.4 / gpt-5.4-mini)

How to make the `codex` CLI hit our Azure OpenAI deployment from both local and the `lsf` GCP VM. Captured 2026-05-27 after debugging the long-standing local 401 issue.

---

## TL;DR — what to export before running codex

The Codex CLI reads `AZURE_OPENAI_API_KEY` from the environment. The key file at `~/api_keys/azure_cloudbank/gpt-54_1.txt` is **YAML, not a raw key** — exporting the whole file gives a bogus 217-char string and Azure returns 401. Extract the value of the `api_key:` line:

```bash
export AZURE_OPENAI_API_KEY=$(awk -F': ' '/^api_key:/{print $2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt)
```

For gpt-5.4-mini use `~/api_keys/azure_cloudbank/gpt-54-mini.txt` instead.

The extracted key should be exactly **84 chars** long. If you ever get a different length, you grabbed the wrong field.

---

## Config file (`~/.codex/config.toml`)

Same on local and server:

```toml
model = "gpt-5.4"
model_provider = "azure"
model_reasoning_effort = "high"

[model_providers.azure]
name = "Azure OpenAI"
base_url = "https://doc-bench.openai.azure.com/openai/v1"
env_key = "AZURE_OPENAI_API_KEY"
wire_api = "responses"
```

Notes:
- The base URL has `/openai/v1` — this is Azure's new "v1 responses" preview endpoint, which Codex CLI defaults to via `wire_api = "responses"`.
- The Python SDK (`src/models/gpt54.py`) uses a *different* shape — `https://doc-bench.openai.azure.com/` + `openai/deployments/<dep>/chat/completions?api-version=…`. Don't try to share endpoints; they each need their own format.
- Provider name `azure` works (it's a built-in reserved name in Codex CLI). The old strategy memo noted this had to be a custom name like `azure-gpt54` — that was wrong; `azure` works.

---

## Invocation

```bash
# Set key once per shell
export AZURE_OPENAI_API_KEY=$(awk -F': ' '/^api_key:/{print $2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt)

# Codex one-shot, non-interactive
codex --ask-for-approval never \
      exec \
      --json --color never \
      --model gpt-5.4 \
      --cd /path/to/repo \
      --sandbox danger-full-access \
      "<prompt>"
```

For gpt-5.4-mini, swap `--model gpt-5.4-mini` and use the mini key file.

---

## Running on the `lsf` server

The GCP VM has codex installed at `~/.npm-global/bin/codex` (npm `@openai/codex@0.131.0` as of writing). `~/.bashrc` exports `~/.npm-global/bin` onto `PATH`, but **non-interactive SSH doesn't source `.bashrc`**, so for `gcloud compute ssh ... --command="..."` you need to export PATH inline:

```bash
gcloud compute ssh lsf --zone=us-central1-a --project=doc-structure --tunnel-through-iap \
  --command="export PATH=\$HOME/.npm-global/bin:\$PATH && \
             export AZURE_OPENAI_API_KEY=\$(awk -F': ' '/^api_key:/{print \$2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt) && \
             cd ~/LSF && codex ... '<prompt>'"
```

`gcloud` itself lives at `/opt/homebrew/share/google-cloud-sdk/bin/gcloud` on the laptop (Homebrew install) — not the Cloud Code installer copy, which is broken. The first thing in any session that calls `gcloud` should be:

```bash
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:"$PATH"
```

---

## Symptoms when the key is wrong

If you see this in codex output:

```
{"type":"error","message":"unexpected status 401 Unauthorized: Access denied due to invalid subscription key or wrong API endpoint…"}
```

Run this and check the length:

```bash
echo "key length: ${#AZURE_OPENAI_API_KEY}"
```

If it's not 84, you exported the wrong field. Re-extract with the `awk` line at the top.

You can sanity-check the key + endpoint with curl, bypassing codex entirely:

```bash
curl -sS -w "\nHTTP %{http_code}\n" \
  -X POST "https://doc-bench.openai.azure.com/openai/v1/responses" \
  -H "api-key: $AZURE_OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","input":"ping"}'
```

A working key returns `HTTP 200` plus a JSON response body. A 401 here means the env var is wrong.

---

## Wrappers in this repo that invoke codex

These all rely on `AZURE_OPENAI_API_KEY` being exported in the calling shell. Export it before invoking them:

- `src/rule_gen/agent_codex.py` — Approach 3 (Agentic-gen) but with Codex as the agent backbone instead of Claude Code
- `src/baseline/agentic_rule_full_data.py` (+ `_gpt54.py`, `_gpt54mini.py` wrappers) — Strategy 4 baseline, one Codex session per question over the full corpus
- `src/baseline/agentic_codex_qa.py` and `agentic_codex_qa_all.py` — per-(question, doc) and full-dataset codex QA baselines
