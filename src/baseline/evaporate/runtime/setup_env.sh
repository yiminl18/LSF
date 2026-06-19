#!/usr/bin/env bash
# Build the isolated Evaporate venv (.venv-evaporate) used by run_variant.py.
#
# Dependency isolation (plan Gap B): the main repo env must NOT carry Evaporate's
# old deps. We install only what `run_variant.py` actually touches in the upstream
# code path it uses (prompts + profiler_utils + profiler's sandboxed function
# executor), PLUS `snorkel` for Code+ weak-supervision aggregation.
#   * Code+ uses the modern Snorkel `LabelModel` (snorkel 0.10) for WS — the
#     maintained successor to upstream's Snorkel-MeTaL `LabelModel` (MeTaL 0.5
#     no longer runs under modern networkx). `--combiner mv` is the fallback.
#   * snorkel-metal / cvxpy / metal are intentionally OMITTED — MeTaL 0.5 crashes
#     on modern networkx (Graph.node removed); reviving it is a dead end. The
#     upstream metal-based `weak_supervision.run_ws` import is still stubbed in
#     run_variant._install_ws_stub (we never import it).
#   * manifest-ml — OMITTED; we route all LLM calls through runtime/llm_backend.py.
#
# Usage:  bash src/baseline/evaporate/runtime/setup_env.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv-evaporate"
PYBIN="${EVAPORATE_PYTHON:-python3}"

echo "[setup_env] creating venv at ${VENV}"
"${PYBIN}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip >/dev/null

echo "[setup_env] installing runtime deps"
# Modern, conflict-free set sufficient for the upstream code we import, plus
# snorkel for Code+ weak supervision (pulls torch/scipy/networkx/tensorboard).
# NOTE: do NOT add snorkel-metal / cvxpy / metal (MeTaL 0.5 is dead on modern
# networkx).
python -m pip install \
  "openai>=1.0,<2" \
  "tiktoken>=0.5" \
  "beautifulsoup4>=4.10" \
  "lxml>=4.9" \
  "pandas>=1.5" \
  "numpy>=1.23" \
  "tqdm>=4.64" \
  "snorkel==0.10.0"

echo "[setup_env] verifying upstream import + sandbox exec (no network)"
python "${HERE}/run_variant.py" --self-test

echo "[setup_env] done. Activate with: source ${VENV}/bin/activate"
