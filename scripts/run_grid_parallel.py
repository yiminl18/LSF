#!/usr/bin/env python3
"""Dependency-parallel driver for the LSF court grid.

Models the pipeline as a DAG and runs independent branches concurrently:

    sampling(s) ─► rule_gen(s,g) ─► precompute(s,g) ─► refine(s,g,r) ─► apply(s,g,r)
                                    (Pareto refiners only)   │
    rule_gen(s,g) ───────────────────────────────────────────┘ (agentic refiner: no precompute)

A node runs as soon as ALL its ancestors have passed their output checks; nodes
on different branches run in parallel (capped by --jobs). After a node runs, its
output is checked; if the check fails the node is marked FAILED and its entire
subtree is PRUNED (skipped) — but sibling branches keep going. This is the tree
rule: "issue a task as soon as its root-to-parent chain is clean."

Idempotent: every pipeline.py call passes --skip-existing, so any work already on
disk (rule pools, caches, prior results) is reused, not recomputed.

Usage:
    python3 scripts/run_grid_parallel.py [--jobs N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup_env() -> None:
    """Make Codex reachable and authenticated for agent_codex / agentic_codex
    stages. The gpt54/gpt54mini Python models read the key from the file
    directly, but the codex CLI needs AZURE_OPENAI_API_KEY in the env and the
    binary on PATH (non-interactive shells don't source ~/.bashrc)."""
    npm_bin = str(Path.home() / ".npm-global" / "bin")
    if npm_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = npm_bin + os.pathsep + os.environ.get("PATH", "")
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        keyfile = Path(os.environ.get("AZURE_KEY_FILE",
                                      Path.home() / "api_keys/azure_cloudbank/gpt-54_1.txt"))
        try:
            for line in keyfile.read_text().splitlines():
                if line.startswith("api_key:"):
                    os.environ["AZURE_OPENAI_API_KEY"] = line.split(":", 1)[1].strip()
                    break
        except FileNotFoundError:
            print(f"WARN: Azure key file not found: {keyfile}", flush=True)
    codex_ok = any((Path(p) / "codex").exists() for p in os.environ["PATH"].split(os.pathsep))
    print(f"env: codex={'found' if codex_ok else 'MISSING'} "
          f"AZURE_OPENAI_API_KEY={'set' if os.environ.get('AZURE_OPENAI_API_KEY') else 'UNSET'}",
          flush=True)

# ── Grid config (court) ─────────────────────────────────────────────────────
DATASET   = "court"
CLUSTER   = "all_docs"
QUERIES   = "data/court/queries.json"
OUTPUT    = Path("results/court/grid")
RULES     = Path(f"rules/{DATASET}/grid")
APPLY     = "default"
PROC      = None   # --processing-dir override; None lets pipeline.py auto-probe
MAX_STAGE = "apply"   # build/run nodes only up to this stage (e.g. "rule_gen")

SAMPLINGS = ["random", "fps"]
RULEGENS  = ["llm_coarse_gpt54", "agent_codex_gpt54"]
PARETO    = ["p_mini", "p_hybrid"]      # need precompute
NONPARETO = ["agentic_codex_gpt54"]     # no precompute
REFINES   = NONPARETO + PARETO

LOGDIR = Path("logs/court_grid_parallel")

# ── Node model ──────────────────────────────────────────────────────────────
# Each node: id tuple, stage (pipeline --stop-after), the (s,g,r) it carries,
# a list of dependency node-ids, and a check (relative paths under ROOT).

def have(dirpath: Path, pattern: str, minimum: int = 1) -> bool:
    p = ROOT / dirpath
    if not p.is_dir():
        return False
    return sum(1 for _ in p.rglob(pattern)) >= minimum


def nonempty(path: Path) -> bool:
    p = ROOT / path
    return p.is_file() and p.stat().st_size > 0


def have_n(dirpath: Path, pattern: str, n: int) -> bool:
    """At least n files matching pattern directly under dirpath (non-recursive)."""
    p = ROOT / dirpath
    if not p.is_dir():
        return False
    return sum(1 for _ in p.glob(pattern)) >= n


def _n_questions() -> int:
    try:
        return len(json.loads((ROOT / QUERIES).read_text()))
    except Exception:  # noqa: BLE001
        return 0


N_QUESTIONS = _n_questions()


def complete_pool(dirpath: Path) -> bool:
    """A rule pool is complete only if it has a populated (>=1 .py) sub-folder
    for every question — not just >=1 .py somewhere. Catches a pool left partial
    by a failed/missing generator (e.g. codex env not set up)."""
    p = ROOT / dirpath
    if not p.is_dir() or N_QUESTIONS == 0:
        return False
    populated = sum(1 for sub in p.iterdir() if sub.is_dir() and any(sub.glob("*.py")))
    return populated >= N_QUESTIONS


_STAGE_ORDER = ["sampling", "rule_gen", "precompute", "refine", "apply"]


def build_graph():
    nodes: dict[tuple, dict] = {}
    _maxrank = _STAGE_ORDER.index(MAX_STAGE)

    def add(nid, stage, s, g, r, deps, check):
        # Skip any node beyond the requested max stage (e.g. --max-stage rule_gen
        # builds only sampling + rule_gen nodes). Deps to skipped nodes never
        # arise because earlier stages are always kept.
        if _STAGE_ORDER.index(stage) > _maxrank:
            return
        nodes[nid] = {"stage": stage, "s": s, "g": g, "r": r, "deps": deps, "check": check}

    for s in SAMPLINGS:
        add(("sampling", s), "sampling", s, RULEGENS[0], "p_mini", [],
            lambda s=s: have(OUTPUT / "sampling" / s, "*.json", 2))

    for s in SAMPLINGS:
        for g in RULEGENS:
            add(("rule_gen", s, g), "rule_gen", s, g, "p_mini", [("sampling", s)],
                lambda s=s, g=g: complete_pool(RULES / s / g))

            add(("precompute", s, g), "precompute", s, g, "p_mini", [("rule_gen", s, g)],
                lambda s=s, g=g: (
                    have(OUTPUT / "cache" / s / g / "cost_profile", "*.json")
                    and have(OUTPUT / "cache" / s / g / "eval_merge_base", "*.json")
                    and have(OUTPUT / "cache" / s / g / "eval_individual_gpt54mini", "*_eval.json")
                ))

            for r in REFINES:
                dep = ("precompute", s, g) if r in PARETO else ("rule_gen", s, g)
                # refine writes <q>_refine.json per question regardless of how many
                # rules it selects, so that's the completeness signal (a question
                # may legitimately select 0 rules → no .py, but still a _refine.json).
                add(("refine", s, g, r), "refine", s, g, r, [dep],
                    lambda s=s, g=g, r=r: have_n(
                        OUTPUT / "refined" / s / g / r, "*_refine.json", N_QUESTIONS))
                # apply writes <q>_unsampled.json per question; require all of them
                # AND the summary — not just a (possibly partial) summary file.
                add(("apply", s, g, r), "apply", s, g, r, [("refine", s, g, r)],
                    lambda s=s, g=g, r=r: (
                        nonempty(OUTPUT / "apply" / s / g / r / APPLY / "pipeline_summary.json")
                        and have_n(OUTPUT / "apply" / s / g / r / APPLY, "*_unsampled.json", N_QUESTIONS)))
    return nodes


def nid_str(nid: tuple) -> str:
    return "/".join(nid)


def run_node(nid: tuple, node: dict) -> tuple[bool, str]:
    """Run pipeline.py up to this node's stage, then check its output.
    Returns (ok, detail). The check is authoritative even if pipeline exits 0."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{node['stage']}__{node['s']}_{node['g']}_{node['r']}.log"
    cmd = [
        sys.executable, "src/pipeline.py",
        "--sampling-strategy", node["s"],
        "--rule-gen-strategy", node["g"],
        "--refine-strategy",   node["r"],
        "--apply-strategy",    APPLY,
        "--queries-file",      QUERIES,
        "--dataset",           DATASET,
        "--cluster",           CLUSTER,
        "--output-dir",        str(OUTPUT),
        "--stop-after",        node["stage"],
        "--skip-existing",
    ]
    if PROC:
        cmd += ["--processing-dir", str(PROC)]
    with open(ROOT / log, "w") as fh:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    ok = node["check"]()
    if not ok:
        return False, f"check failed (rc={rc}, log={log})"
    return True, f"ok (log={log})"


def main():
    global DATASET, CLUSTER, QUERIES, OUTPUT, RULES, PROC, LOGDIR, N_QUESTIONS, MAX_STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="max concurrent pipeline.py processes")
    ap.add_argument("--dry-run", action="store_true", help="print the DAG and exit")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--output",  default=str(OUTPUT))
    ap.add_argument("--rules",   default=str(RULES))
    ap.add_argument("--proc-dir", default=None, help="--processing-dir for pipeline.py (e.g. data/nopv/json)")
    ap.add_argument("--max-stage", default="apply", choices=_STAGE_ORDER,
                    help="run nodes only up to this stage (e.g. rule_gen = sampling+rule_gen only)")
    args = ap.parse_args()

    MAX_STAGE = args.max_stage
    DATASET, CLUSTER, QUERIES = args.dataset, args.cluster, args.queries
    OUTPUT, RULES, PROC = Path(args.output), Path(args.rules), args.proc_dir
    LOGDIR = Path(f"logs/{DATASET}_grid_parallel")
    N_QUESTIONS = _n_questions()

    setup_env()
    nodes = build_graph()
    order = list(nodes.keys())

    if args.dry_run:
        print(f"DAG: {len(nodes)} nodes, jobs={args.jobs}\n")
        for nid in order:
            deps = nodes[nid]["deps"]
            print(f"  {nid_str(nid):45s} <- {', '.join(nid_str(d) for d in deps) or '(root)'}")
        return 0

    status = {nid: "pending" for nid in order}   # pending|running|ok|failed|skipped
    t0 = time.time()

    def log(msg):
        print(f"[{int(time.time()-t0):5d}s] {msg}", flush=True)

    log(f"start: {len(nodes)} nodes, jobs={args.jobs}")
    futures = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        while any(status[n] in ("pending", "running") for n in order):
            # 1) prune: any pending node with a failed/skipped ancestor
            for n in order:
                if status[n] == "pending" and any(
                        status[d] in ("failed", "skipped") for d in nodes[n]["deps"]):
                    status[n] = "skipped"
                    log(f"PRUNE  {nid_str(n)} (ancestor failed)")
            # 2) submit: pending nodes whose deps are all ok
            for n in order:
                if status[n] == "pending" and all(status[d] == "ok" for d in nodes[n]["deps"]):
                    status[n] = "running"
                    log(f"RUN    {nid_str(n)}")
                    futures[ex.submit(run_node, n, nodes[n])] = n
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                n = futures.pop(fut)
                try:
                    ok, detail = fut.result()
                except Exception as e:  # noqa: BLE001
                    ok, detail = False, f"exception: {e}"
                status[n] = "ok" if ok else "failed"
                log(f"{'OK   ' if ok else 'FAIL '} {nid_str(n)} — {detail}")

    # ── Summary ──
    print("\n" + "=" * 74)
    counts = {k: sum(1 for v in status.values() if v == k) for k in ("ok", "failed", "skipped")}
    print(f"DONE in {int(time.time()-t0)}s — ok={counts['ok']} failed={counts['failed']} "
          f"skipped={counts['skipped']}")
    bad = [nid_str(n) for n in order if status[n] in ("failed", "skipped")]
    if bad:
        print("Pruned / failed branches:")
        for b in bad:
            print(f"  ⚠️  {b} ({status[next(n for n in order if nid_str(n)==b)]})")
        print("=" * 74)
        return 2
    print("✓ all branches completed clean")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
