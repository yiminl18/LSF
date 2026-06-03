"""End-to-end LSF pipeline: sampling -> rule gen -> (optional) refine -> apply + eval.

Each of the four stages exposes a strategy choice that maps to a documented
approach in docs/approach/*.md. See docs/pipeline.md for the full design.

Usage:

    from pipeline import run_pipeline

    result = run_pipeline(
        sampling_strategy = "random",
        rule_gen_strategy = "llm_coarse",
        refine_strategy   = "p_mini",
        apply_strategy    = "default",
        dataset           = "financebench",
        cluster           = "single_cluster",
        output_dir        = "results/e2e",
    )

Or via CLI:

    python src/pipeline.py --rule-gen-strategy llm_coarse --refine-strategy p_mini --apply-strategy default
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Strategy registries ─────────────────────────────────────────────────────

SAMPLING_STRATEGIES = ("random", "fps")
RULE_GEN_STRATEGIES = (
    "llm_coarse", "llm_coarse_gpt54", "llm_coarse_gpt54mini",
    "agent_langchain",
    "agent_claude",
    "agent_codex", "agent_codex_gpt54", "agent_codex_gpt54mini",
)
REFINE_STRATEGIES   = (
    "none",
    "v1", "p_mini", "p_gpt54", "p_proxy", "p_v2", "p_v3", "p_hybrid",
    "agentic",
    "agentic_codex", "agentic_codex_gpt54", "agentic_codex_gpt54mini",
)
APPLY_STRATEGIES    = ("merge", "default")

# Suffix → model alias. Strategies ending in `_gpt54` / `_gpt54mini` are model-explicit
# variants of an underlying base strategy. The base strategy (no suffix) defaults to gpt54.
_MODEL_SUFFIX = {"_gpt54": "gpt54", "_gpt54mini": "gpt54mini"}


def _split_strategy_model(strategy: str, default_model: str) -> tuple[str, str]:
    """Split 'foo_gpt54mini' -> ('foo', 'gpt54mini'). Returns (base, model)."""
    for suffix, model in _MODEL_SUFFIX.items():
        if strategy.endswith(suffix):
            return strategy[: -len(suffix)], model
    return strategy, default_model


# ── Small helpers ───────────────────────────────────────────────────────────

def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# Filename patterns we try when locating a reconstructed-JSON for a doc.
# FinanceBench uses `<DOC>_reconstructed.json`; court uses `<DOC>.json`.
_DOC_JSON_CANDIDATES = ("{stem}_reconstructed.json", "{stem}.json")


def _load_docs(labels: dict, processing_dir: str) -> dict[str, dict]:
    """Load reconstructed-JSON docs for the keys in `labels`.

    Tries `<DOC>_reconstructed.json` first (FinanceBench convention), falls back
    to `<DOC>.json` (court convention). Either layout works.
    """
    docs: dict[str, dict] = {}
    pdir = Path(processing_dir)
    for pdf_key in labels:
        doc_name = pdf_key.replace(".pdf", "").replace(".PDF", "")
        loaded = False
        for tmpl in _DOC_JSON_CANDIDATES:
            path = pdir / tmpl.format(stem=doc_name)
            if path.exists():
                docs[doc_name] = json.loads(path.read_text(encoding="utf-8"))
                loaded = True
                break
        if not loaded:
            tried = ", ".join(tmpl.format(stem=doc_name) for tmpl in _DOC_JSON_CANDIDATES)
            print(f"  WARN: missing doc JSON for {doc_name!r} in {pdir} (tried: {tried})", flush=True)
    return docs


def _default_processing_dir(dataset: str) -> Path:
    """Return the conventional processing dir for a dataset, probing in order:
    `data/<dataset>/processing` (FinanceBench), then `data/<dataset>/json` (court).
    """
    for sub in ("processing", "json"):
        cand = _ROOT / "data" / dataset / sub
        if cand.is_dir():
            return cand
    return _ROOT / "data" / dataset / "processing"   # fallback (will fail later with a clear error)


def _list_rule_names(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.startswith("rule_") and f.endswith(".py")
    )


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_SAMPLE_CAP = 20  # hard cap from pipeline.md


def _load_queries(queries_file: str) -> list[str]:
    """Read questions from a plain-text file (one per line) OR a JSON file
    that is either ['q1', 'q2', ...] OR [{'text': 'q1', ...}, ...]. Returns
    a list of plain question strings.
    """
    p = Path(queries_file)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [q["text"] for q in data]
        if isinstance(data, list):
            return [str(q) for q in data]
        raise ValueError(f"Unsupported queries.json shape in {queries_file}")
    return [l.strip() for l in raw.splitlines() if l.strip()]


def _build_random_split(all_labels_path: Path, output_dir: Path, cap: int, seed: int = 0) -> tuple[Path, Path]:
    """Build a random sample/unsampled split from a flat all_labels.json (court/nopv style).

    Writes the two label files into output_dir and returns their paths.
    """
    import random as _random
    labs = json.loads(all_labels_path.read_text(encoding="utf-8"))
    keys = sorted(labs)
    rng  = _random.Random(seed)
    rng.shuffle(keys)
    k = min(cap, len(keys))
    samp, unsamp = keys[:k], keys[k:]
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = output_dir / "sample_doc_labels.json"
    up = output_dir / "unsampled_doc_labels.json"
    _write_json(sp, {k: labs[k] for k in samp})
    _write_json(up, {k: labs[k] for k in unsamp})
    print(f"  [sampling:random] built {k}/{len(unsamp)} split from {all_labels_path}", flush=True)
    return sp, up


# ── Stage 1 — Sampling ──────────────────────────────────────────────────────

def stage_sampling(
    *,
    strategy: str,
    dataset: str,
    cluster: str,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[Path, Path]:
    """Return (sample_labels_path, unsampled_labels_path).

    For FinanceBench we read pre-built `data/financebench/sample/<cluster>/<sampler>/` JSONs.
    For court/nopv/officeqa we derive the split at run-time from `data/<dataset>/all_labels.json`.
    """
    base       = _ROOT / "data" / dataset / "sample" / cluster
    all_labels = _ROOT / "data" / dataset / "all_labels.json"
    grid_dir   = output_dir / "sampling" / strategy

    if strategy == "random":
        # Preferred path: a pre-built FinanceBench split.
        sample    = base / "random" / "sample_doc_labels.json"
        unsampled = base / "random" / "unsampled_doc_labels.json"
        if sample.exists() and unsampled.exists():
            return sample, unsampled

        # Fallback: derive split at run-time from a flat all_labels.json (court/nopv/officeqa).
        if all_labels.exists():
            sp = grid_dir / "sample_doc_labels.json"
            up = grid_dir / "unsampled_doc_labels.json"
            if skip_existing and sp.exists() and up.exists():
                print(f"  [sampling:random] SKIP (exists at {grid_dir})", flush=True)
                return sp, up
            return _build_random_split(all_labels, grid_dir, cap=_SAMPLE_CAP)

        raise FileNotFoundError(
            f"random sample: no pre-built split at {base/'random'} and no "
            f"{all_labels} to build from."
        )

    if strategy == "fps":
        # Default to grid layout under output_dir; legacy path under data/<dataset>/sample is also accepted.
        sp = grid_dir / "sample_doc_labels.json"
        up = grid_dir / "unsampled_doc_labels.json"
        if skip_existing and sp.exists() and up.exists():
            print(f"  [sampling:fps] SKIP (exists at {grid_dir})", flush=True)
            return sp, up

        # The FPS driver currently only supports FinanceBench out of the box. For other datasets
        # we pass through the env via CLI args once run_fps_sampling.py is parameterised.
        print(f"  [sampling:fps] running src/sampling/run_fps_sampling.py for {dataset}…", flush=True)
        cmd = [
            sys.executable, "src/sampling/run_fps_sampling.py",
            "--dataset", dataset,
            "--max-K", str(_SAMPLE_CAP),
            "--output-dir", str(grid_dir),
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            # Fall back to legacy hardcoded behavior (FinanceBench only) if the script
            # doesn't yet support these args.
            print("  [sampling:fps] driver did not accept new CLI; falling back to legacy invocation", flush=True)
            proc = subprocess.run(
                [sys.executable, "src/sampling/run_fps_sampling.py"],
                cwd=str(_ROOT), capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"FPS sampling failed:\n{proc.stderr[-2000:]}")
            legacy_dir = base / "fps"
            return (legacy_dir / "sample_doc_labels.json",
                    legacy_dir / "unsampled_doc_labels.json")
        if not sp.exists() or not up.exists():
            raise FileNotFoundError(f"FPS finished but label files not at {grid_dir}")
        return sp, up

    raise ValueError(f"Unknown sampling_strategy: {strategy!r}")


# ── Stage 2 — Rule Generation ──────────────────────────────────────────────

# Subprocess-based generators (agent_claude / agent_codex) operate per-question
# and don't expose the same callable interface as llm_coarse / agent_langchain.
_SUBPROCESS_GEN = {"agent_claude", "agent_codex"}


def _write_rule_gen_stats(
    *,
    output_dir: Path,
    rules_dir: Path,
    strategy: str,
    model: str,
    question: str,
    question_slug: str,
    rule_folder: Path,
    latency_seconds: float,
    raw_meta: dict,
) -> None:
    """Persist a normalized per-(sampling, rule_gen, question) stats record.

    Path: <output_dir>/rule_gen/<sampling>/<rule_gen>/<q_slug>.json

    Same schema regardless of which underlying rule_gen strategy ran, so the grid
    summary can roll them up without per-strategy parsing.
    """
    # Derive sampling strategy from rules_dir (which is .../<sampling>/<rule_gen>)
    try:
        sampling_strategy = rules_dir.parent.name
        rule_gen_strategy = rules_dir.name
    except Exception:
        sampling_strategy, rule_gen_strategy = "unknown", strategy

    rules = sorted(p.stem for p in rule_folder.glob("rule_*.py"))

    # llm_coarse returns rich metadata directly; subprocess gen returns minimal info.
    in_tok  = int(raw_meta.get("input_tokens",  0) or 0)
    out_tok = int(raw_meta.get("output_tokens", 0) or 0)

    # If subprocess gen wrote its own *_rule_gen.json next to the rule pool, harvest tokens.
    if in_tok == 0 and out_tok == 0:
        candidate = sorted(rule_folder.parent.glob(f"{question_slug}*_rule_gen.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidate:
            try:
                d = json.loads(c.read_text())
                in_tok  = int(d.get("input_tokens",  d.get("agent_input_tokens",  0)) or 0)
                out_tok = int(d.get("output_tokens", d.get("agent_output_tokens", 0)) or 0)
                if in_tok or out_tok:
                    break
            except Exception:
                pass

    # Rough cost estimate at gpt-5.4 standard rates ($1.25/M in, $10/M out).
    # gpt-5.4-mini would be ~10x cheaper; we report the gpt-5.4 ceiling.
    cost_usd = round(in_tok / 1e6 * 1.25 + out_tok / 1e6 * 10.0, 4)

    stats = {
        "question":           question,
        "question_slug":      question_slug,
        "sampling_strategy":  sampling_strategy,
        "rule_gen_strategy":  rule_gen_strategy,
        "strategy_full":      strategy,
        "model":              model,
        "n_rules":            len(rules),
        "rules":              rules,
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "latency_seconds":    round(latency_seconds, 2),
        "approx_cost_usd_gpt54_rate": cost_usd,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }
    stats_path = output_dir / "rule_gen" / sampling_strategy / rule_gen_strategy / f"{question_slug}.json"
    _write_json(stats_path, stats)


def stage_rule_gen(
    *,
    strategy: str,
    question: str,
    question_slug: str,
    sample_docs: list[dict],
    sample_doc_names: list[str],
    ground_truth: dict[str, Any],
    rules_dir: Path,
    output_dir: Path,
    model: str,
    skip_existing: bool,
) -> tuple[Path, dict]:
    """Generate rules for one question. Return (rule_folder, gen_metadata).

    Side effect: writes a normalized stats record to
    <output_dir>/rule_gen/<sampling>/<rule_gen>/<q_slug>.json.
    """

    # Strip _gpt54 / _gpt54mini suffix and override the model arg
    base_strategy, effective_model = _split_strategy_model(strategy, default_model=model)

    # rules_dir already encodes (dataset, sampling, rule_gen). The leaf is the pure slug.
    rule_folder = rules_dir / question_slug
    rule_gen_out = rule_folder.parent / f"{question_slug}_rule_gen.json"

    if skip_existing and rule_folder.is_dir() and _list_rule_names(rule_folder):
        print(f"  [gen:{strategy}] SKIP {question_slug}", flush=True)
        return rule_folder, _read_json(rule_gen_out, default={})

    rule_folder.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result: dict = {}

    if base_strategy in {"llm_coarse", "agent_langchain"}:
        mod = importlib.import_module(f"rule_gen.{base_strategy}")
        fn  = next(v for k, v in vars(mod).items() if k.startswith("rule_gen_") and callable(v))
        result = fn(
            documents     = sample_docs,
            question      = question,
            ground_truth  = ground_truth,
            rules_dir     = str(rules_dir),
            output_dir    = str(output_dir / "rule_gen"),
            model_name    = effective_model,
            rule_subdir   = str(rule_folder),
        )
        _write_json(rule_gen_out, result)
        print(f"  [gen:{strategy} model={effective_model}] {question_slug}: {len(result.get('rules', []))} rules", flush=True)
    elif base_strategy in _SUBPROCESS_GEN:
        mod_name = "agent_claude" if base_strategy == "agent_claude" else "agent_codex"
        cmd = [
            sys.executable, f"src/rule_gen/{mod_name}.py",
            question,
            "--docs", *sample_doc_names,
            "--rules-dir", str(rules_dir),
            "--model", effective_model,
            "--question-slug", question_slug,
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{strategy} failed for {question_slug}:\n{proc.stderr[-2000:]}")
        result = {"strategy": strategy, "model": effective_model, "stdout_tail": proc.stdout[-500:]}
        _write_json(rule_gen_out, result)
        print(f"  [gen:{strategy} model={effective_model}] {question_slug}: done", flush=True)
    else:
        raise ValueError(f"Unknown rule_gen_strategy: {strategy!r}")

    latency_s = time.time() - t0

    # Normalized per-(sampling, rule_gen, question) stats — predictable path, same schema for every strategy.
    try:
        _write_rule_gen_stats(
            output_dir=output_dir, rules_dir=rules_dir,
            strategy=strategy, model=effective_model,
            question=question, question_slug=question_slug,
            rule_folder=rule_folder, latency_seconds=latency_s, raw_meta=result,
        )
    except Exception as e:
        print(f"  WARN: rule_gen stats writer failed: {e}", flush=True)

    return rule_folder, result


# ── Phase C — Per-pool precompute (cost_profile, eval_individual, eval_merge_base) ──

# Only refine strategies that read precomputed caches need this.
_REFINE_NEEDS_PRECOMPUTE = {"p_mini", "p_gpt54", "p_hybrid", "p_v2", "p_v3"}


def stage_precompute(
    *,
    rule_folder: Path,
    cache_root: Path,
    sample_docs: dict[str, dict],
    sample_labels: dict[str, dict],
    question: str,
    question_slug: str,
    rules_dir: Path,
    needed_judges: tuple[str, ...] = ("gpt54mini",),
    skip_existing: bool = True,
) -> None:
    """Build per-question caches needed by Pareto-family refine strategies.

    Model policy: COVERAGE is estimated cheaply with gpt54mini; the BASELINE /
    target-doc set (verify) is anchored with gpt54.
      cost_profile/<q_slug>.json         per-rule avg_cost_ratio. FREE — uses
                                         rule_apply_merge(retrieve_only=True), so
                                         no LLM call (only retrieved-token count).
      eval_individual_gpt54mini/<q_slug>/<rule_name>_eval.json  per-rule coverage:
                                         answer + judge both gpt54mini (cheap).
      eval_merge_base/<q_slug>.json      full-pool merge accuracy on sampled docs;
                                         answer + judge gpt54 (defines D*).
    """
    from rule_apply.merge       import rule_apply_merge
    from rule_refine.selection.eval_judge import judge as _judge_fn

    rule_names = _list_rule_names(rule_folder)
    if not rule_names:
        print(f"  [precompute] no rules in {rule_folder}; skipping", flush=True)
        return

    # ── 1. cost_profile (free) ─────────────────────────────────────────────
    cost_out = cache_root / "cost_profile" / f"{question_slug}.json"
    if skip_existing and cost_out.exists():
        print(f"  [precompute:cost] SKIP {question_slug}", flush=True)
    else:
        profile: dict[str, dict] = {}
        for r in rule_names:
            costs = []
            for d in sample_docs.values():
                try:
                    res = rule_apply_merge(
                        document=d, rule_names=[r],
                        question_slug=question_slug, question=question,
                        rules_dir=str(rules_dir),
                        output_dir=str(cache_root / "_cost_tmp"),
                        retrieve_only=True,   # free: no LLM call, only token count
                    )
                    total_tok = _count_tokens("\n".join(s.get("text","") for s in d.get("texts",[])))
                    ret_tok   = res.get("retrieved_token_count", 0) or 0
                    costs.append(ret_tok / total_tok if total_tok else 0.0)
                except Exception as e:
                    print(f"    cost {r}: {e}")
            profile[r] = {"avg_cost_ratio": (sum(costs)/len(costs)) if costs else 0.0}
        _write_json(cost_out, profile)
        print(f"  [precompute:cost] {question_slug}: {len(profile)} rules profiled", flush=True)

    # ── 2. eval_individual — per-rule COVERAGE, gpt54mini answer + judge ──
    for judge_model in needed_judges:
        ev_dir = cache_root / f"eval_individual_{judge_model}" / question_slug
        if skip_existing and ev_dir.is_dir() and len(list(ev_dir.glob("*_eval.json"))) >= len(rule_names):
            print(f"  [precompute:ev_{judge_model}] SKIP {question_slug}", flush=True)
            continue
        ev_dir.mkdir(parents=True, exist_ok=True)
        for r in rule_names:
            out_path = ev_dir / f"{r}_eval.json"
            if skip_existing and out_path.exists():
                continue
            n_correct = 0
            for doc_name, doc in sample_docs.items():
                try:
                    res = rule_apply_merge(
                        document=doc, rule_names=[r],
                        question_slug=question_slug, question=question,
                        model_name="gpt54mini", rules_dir=str(rules_dir),
                        output_dir=str(cache_root / "_ev_tmp"),
                    )
                    gt = sample_labels.get(doc_name + ".pdf", {}).get(question)
                    ok, _, _ = _judge_fn(question, gt, res.get("predicted_answer"), model_name=judge_model)
                    if ok:
                        n_correct += 1
                except Exception as e:
                    print(f"    eval_ind {r}/{doc_name}: {e}")
            _write_json(out_path, {
                "rule_name": r, "judge_model": judge_model,
                "accuracy": n_correct / len(sample_docs) if sample_docs else 0.0,
                "n_correct": n_correct, "n": len(sample_docs),
            })
        print(f"  [precompute:ev_{judge_model}] {question_slug}: {len(rule_names)} rules evaluated", flush=True)

    # ── 3. eval_merge_base (paid; defines D*) ──────────────────────────────
    merge_out = cache_root / "eval_merge_base" / f"{question_slug}.json"
    if skip_existing and merge_out.exists():
        print(f"  [precompute:merge_base] SKIP {question_slug}", flush=True)
    else:
        per_doc: list[dict] = []
        for doc_name, doc in sample_docs.items():
            try:
                res = rule_apply_merge(
                    document=doc, rule_names=rule_names,
                    question_slug=question_slug, question=question,
                    model_name="gpt54", rules_dir=str(rules_dir),
                    output_dir=str(cache_root / "_merge_tmp"),
                )
                gt = sample_labels.get(doc_name + ".pdf", {}).get(question)
                ok, _, _ = _judge_fn(question, gt, res.get("predicted_answer"), model_name="gpt54")
                per_doc.append({"doc_name": doc_name, "correct": ok})
            except Exception as e:
                print(f"    merge_base {doc_name}: {e}")
                per_doc.append({"doc_name": doc_name, "correct": False, "error": str(e)})
        n_correct = sum(1 for r in per_doc if r["correct"])
        _write_json(merge_out, {
            "question": question, "question_slug": question_slug,
            "n": len(per_doc), "n_correct": n_correct,
            "accuracy": n_correct / len(per_doc) if per_doc else 0.0,
            "target_docs": [r["doc_name"] for r in per_doc if r["correct"]],
            "per_doc": per_doc,
        })
        print(f"  [precompute:merge_base] {question_slug}: acc={n_correct}/{len(per_doc)}", flush=True)


# ── Stage 3 — Rule Refinement ──────────────────────────────────────────────

def _materialize_selected(result: dict, rule_folder: Path, refined_folder: Path) -> int:
    """Copy the selected rules' .py files from the source pool into refined_folder.

    Pareto-family selectors return a `selected_rules` list but do NOT write the
    rule files themselves; Stage 4 reads .py files from refined_folder, so we
    materialize the selection here (mirrors the agentic branch's copy step)."""
    import shutil
    chosen = result.get("selected_rules") or []
    if not chosen:
        # The Pareto greedy cover can admit nothing on a small, COMPLEMENTARY pool
        # (each rule retrieves a fragment, none covers a doc alone) even when the
        # merged pool scores well. Never ship an empty refined set: fall back to
        # the full Stage-2 pool — at worst, refinement is a no-op, not a failure.
        chosen = _list_rule_names(rule_folder)
        result["selected_rules"] = chosen
        result["fallback_full_pool"] = True
        print(f"  [refine] selection empty → fallback to full pool ({len(chosen)} rules)", flush=True)
    n = 0
    for name in chosen:
        src = rule_folder / f"{name}.py"
        if src.exists():
            shutil.copy2(src, refined_folder / f"{name}.py")
            n += 1
        else:
            print(f"  WARN: selected {name!r} but {src} missing", flush=True)
    print(f"  [refine] materialized {n}/{len(chosen)} rule(s) → {refined_folder}", flush=True)
    return n


def stage_refine(
    *,
    strategy: str,
    question: str,
    question_slug: str,
    rule_folder: Path,
    sample_docs: list[dict],
    ground_truth: dict[str, Any],
    rules_dir: Path,
    refined_root: Path,            # <out>/refined/<sampling>/<rule_gen>/<refine>/
    cache_root:   Path | None = None,  # <out>/cache/<sampling>/<rule_gen>/ (for p_hybrid etc.)
    sample_labels_path: Path | None = None,  # sampling JSON, threaded to agentic_codex prompt
    processing_dir:     str  | None = None,  # doc-JSON dir, threaded to agentic_codex prompt
    model: str = "gpt54",
    skip_existing: bool = False,
) -> Path:
    """Run refinement and return the *refined* rule folder. Caller passes this folder to Stage 4."""

    refined_dir    = refined_root
    refined_folder = refined_root / question_slug
    refine_out     = refined_root / f"{question_slug}_refine.json"

    if skip_existing and refined_folder.is_dir() and _list_rule_names(refined_folder):
        print(f"  [refine:{strategy}] SKIP {question_slug}", flush=True)
        return refined_folder

    refined_folder.mkdir(parents=True, exist_ok=True)
    rule_names = _list_rule_names(rule_folder)

    if strategy == "v1":
        from rule_refine.v1 import rule_refine as _v1, evaluate_merge_accuracy
        model_mod = importlib.import_module(f"models.{model}")
        target_acc, *_ = evaluate_merge_accuracy(
            rule_names=rule_names, documents=sample_docs,
            ground_truth=ground_truth, question=question,
            rule_folder=rule_folder, model_mod=model_mod,
        )
        result = _v1(
            rule_names=rule_names, target_accuracy=target_acc,
            question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _write_json(refine_out, result)
        return refined_folder

    if strategy in {"p_mini", "p_gpt54"}:
        # p_mini selects with gpt54mini; p_gpt54 with gpt54. Reads precompute caches.
        from rule_refine.selection.select_rules_pareto import run_selection_pareto
        if cache_root is None:
            raise ValueError("p_mini/p_gpt54 need cache_root for cost_profile + eval caches")
        sel_model = "gpt54mini" if strategy == "p_mini" else "gpt54"
        cost_profile = _read_json(cache_root / "cost_profile" / f"{question_slug}.json", default={}) or {}
        result = run_selection_pareto(
            rules_dir=str(rules_dir),
            eval_merge_path=cache_root / "eval_merge_base" / f"{question_slug}.json",
            eval_individual_dir=cache_root / f"eval_individual_{sel_model}" / question_slug,
            documents={d.get("doc_name", str(i)): d for i, d in enumerate(sample_docs)},
            question_slug=question_slug, question=question,
            labels=ground_truth, cost_profile=cost_profile,
            model_name=sel_model, output_dir=str(refined_dir),
        )
        _materialize_selected(result, rule_folder, refined_folder)
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_proxy":
        from rule_refine.selection.select_rules_pareto_proxy import select_rules_pareto_proxy
        result = select_rules_pareto_proxy(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _materialize_selected(result, rule_folder, refined_folder)
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_v2":
        from rule_refine.selection.select_rules_pareto_v2 import select_rules_pareto_v2
        result = select_rules_pareto_v2(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _materialize_selected(result, rule_folder, refined_folder)
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_v3":
        from rule_refine.selection.select_rules_pareto_v3 import select_rules_pareto_v3
        result = select_rules_pareto_v3(
            rule_names=rule_names, question=question, question_slug=question_slug,
            documents=sample_docs, ground_truth=ground_truth,
            rules_dir=str(rules_dir), output_dir=str(refined_dir),
        )
        _materialize_selected(result, rule_folder, refined_folder)
        _write_json(refine_out, result)
        return refined_folder

    if strategy == "p_hybrid":
        from rule_refine.selection.select_rules_pareto_hybrid import run_selection_pareto_hybrid
        if cache_root is None:
            raise ValueError("p_hybrid needs cache_root for cost_profile + eval_individual caches")
        cost_profile_path = cache_root / "cost_profile" / f"{question_slug}.json"
        eval_merge_path   = cache_root / "eval_merge_base" / f"{question_slug}.json"
        ev_ind_gpt54      = cache_root / "eval_individual_gpt54" / question_slug
        ev_ind_mini       = cache_root / "eval_individual_gpt54mini" / question_slug
        cost_profile = _read_json(cost_profile_path, default={}) or {}
        docs_map = {d.get("doc_name", str(i)): d for i, d in enumerate(sample_docs)}
        result = run_selection_pareto_hybrid(
            rules_dir=str(rules_dir), question_slug=question_slug, question=question,
            documents=docs_map, labels=ground_truth,
            eval_individual_dir=ev_ind_gpt54,            # gpt54-judged (for verify-floor)
            eval_merge_path=eval_merge_path,             # base accuracy on full pool
            cost_profile=cost_profile,
            output_dir=str(refined_dir),
            coverage_eval_individual_dir=ev_ind_mini,    # mini-judged (used as sort key)
            coverage_model="gpt54mini",
            verify_model="gpt54",
        )
        _materialize_selected(result, rule_folder, refined_folder)
        _write_json(refine_out, result)
        return refined_folder

    base_strategy, effective_model = _split_strategy_model(strategy, default_model=model)
    if base_strategy in {"agentic", "agentic_codex"}:
        # Driver script handles per-question subprocess + trace capture.
        driver = ("src/rule_refine/agentic.py"
                  if base_strategy == "agentic"
                  else "src/rule_refine/agentic_codex.py")
        cmd = [
            sys.executable, driver,
            "--slug", question_slug,
            "--out-dir", str(refined_dir),
        ]
        if base_strategy == "agentic_codex":
            # Pass explicit question + all path overrides; the codex driver bakes them into the prompt
            # so the agent invokes helper tools (verify_accuracy, compute_cost, etc.) with the right paths.
            cmd += [
                "--model", effective_model,
                "--question", question,
                "--rules-dir",      str(rule_folder),
                "--trace-dir",      str(refined_dir / "_trace"),
                "--selector-run-dir", str(refined_dir / "_selector_run"),
            ]
            if sample_labels_path is not None:
                cmd += ["--sampled-labels", str(sample_labels_path)]
            if processing_dir is not None:
                cmd += ["--processing-dir", str(processing_dir)]
            if cache_root is not None:
                cmd += [
                    "--cost-cache-dir",   str(cache_root / "cost_profile"),
                    "--cov-cache-dir",    str(cache_root / "eval_individual_gpt54"),
                    "--eval-merge-dir",   str(cache_root / "eval_merge_base"),
                ]
        else:
            # Legacy claude driver keeps single --rules-dir flag
            cmd += ["--rules-dir", str(rules_dir)]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{strategy} refine failed for {question_slug}:\n{proc.stderr[-2000:]}")

        # Both agentic drivers write their selection JSON as <out-dir>/<slug>.json
        # (a list of selected rule names). Stage 4 reads .py files from
        # refined_folder = <out-dir>/<slug>/, so materialise the selection by
        # copying the chosen .py files out of the rule pool.
        selection_json = refined_dir / f"{question_slug}.json"
        sel = (_read_json(selection_json, default={}) or {}) if selection_json.exists() else {}
        chosen = sel.get("selected_rules") or []
        if not chosen:
            # Agent selected nothing (e.g. it misjudged the labels) or wrote no
            # selection — never ship an empty refined set: fall back to the full
            # Step-2 pool, same floor as the Pareto refiners (_materialize_selected).
            # Worst case refinement is a no-op, not a silently dropped question.
            chosen = _list_rule_names(rule_folder)
            print(f"  [refine:{strategy}] empty selection → fallback to full pool ({len(chosen)} rules)", flush=True)
        import shutil
        for name in chosen:
            src = rule_folder / f"{name}.py"
            dst = refined_folder / f"{name}.py"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  WARN: agentic refine selected {name!r} but {src} missing", flush=True)
        print(f"  [refine:{strategy}] copied {len(chosen)} rule(s) → {refined_folder}", flush=True)

        _write_json(refine_out, {
            "strategy": strategy, "model": effective_model,
            "stdout_tail": proc.stdout[-500:],
        })
        return refined_folder

    raise ValueError(f"Unknown refine_strategy: {strategy!r}")


# ── Stage 4 — Rule Application + Evaluation ────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an answer equivalence judge. You will be given a question, a "
    "predicted answer, and a ground truth answer. Judge whether the predicted "
    "answer is semantically equivalent to the ground truth.\n\n"
    "Reply with exactly one word: CORRECT or INCORRECT"
)


def _judge(model_mod, question: str, ground_truth, predicted) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10, temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip().lower() == "correct"


def stage_apply_and_eval(
    *,
    apply_strategy: str,
    question: str,
    question_slug: str,
    rule_folder: Path,            # the rule set to apply (refined if Stage 3 ran, else Stage 2 pool)
    fallback_folder: Path | None, # full pool, only needed for apply_strategy="default"
    sample_docs: dict[str, dict],
    unsampled_docs: dict[str, dict],
    sample_labels: dict[str, dict],
    unsampled_labels: dict[str, dict],
    rules_dir: Path,
    apply_root: Path,             # <out>/apply/<sampling>/<rule_gen>/<refine>/<apply>/
    model: str = "gpt54",
    skip_existing: bool = False,
) -> dict[str, dict]:
    """Return {'sampled': eval_dict, 'unsampled': eval_dict}."""

    rule_names = _list_rule_names(rule_folder)
    if not rule_names:
        raise ValueError(f"No rules in {rule_folder}")
    rule_set_slug = "__".join(sorted(rule_names))[:120]

    eval_results: dict[str, dict] = {}
    model_mod = importlib.import_module(f"models.{model}")

    splits = [
        ("sampled",   sample_docs,    sample_labels),
        ("unsampled", unsampled_docs, unsampled_labels),
    ]

    for split, doc_map, labels_dict in splits:
        run_dir  = apply_root / question_slug / split
        run_file = run_dir / f"{rule_set_slug}.json"
        eval_out = apply_root / f"{question_slug}_{split}.json"

        # Resume support: if run_file already has records, skip those docs and append the rest.
        existing_records = _read_json(run_file, default=[])
        if not isinstance(existing_records, list):
            existing_records = []
        already = {r.get("doc_name", "") for r in existing_records}

        if skip_existing and already.issuperset(doc_map.keys()) and eval_out.exists():
            print(f"  [apply+eval:{split}] SKIP", flush=True)
            ed = _read_json(eval_out, default={})
            eval_results[split] = {k: v for k, v in ed.items() if k != "per_doc"}
            continue

        # ── Apply ──
        run_dir.mkdir(parents=True, exist_ok=True)
        remaining = {dn: d for dn, d in doc_map.items() if dn not in already}
        print(f"  [apply:{split}] {apply_strategy}  rules={len(rule_names)}  docs={len(remaining)}", flush=True)

        records = list(existing_records)   # accumulate as we go
        for doc_name, document in remaining.items():
            try:
                rec = _apply_one(
                    apply_strategy   = apply_strategy,
                    document         = document,
                    rule_names       = rule_names,
                    rule_folder      = rule_folder,
                    fallback_folder  = fallback_folder,
                    question         = question,
                    question_slug    = question_slug,
                    rules_dir        = rules_dir,
                    output_dir       = run_dir.parent,    # `apply_root/question_slug/`
                    model            = model,
                )
                # Normalise to per-doc record shape the evaluator expects.
                if isinstance(rec, dict):
                    rec = dict(rec)
                    rec.setdefault("doc_name", doc_name)
                    records.append(rec)
            except Exception as e:
                print(f"    ERROR apply {doc_name}: {e}", flush=True)
                records.append({"doc_name": doc_name, "predicted_answer": None, "error": str(e)})
            # Flush after every doc so external progress watchers see live counts.
            _write_json(run_file, records)

        # ── Evaluate ──
        preds   = {r["doc_name"]: r for r in records}
        per_doc: list[dict] = []
        for doc_name, document in doc_map.items():
            pred_rec  = preds.get(doc_name, {})
            predicted = pred_rec.get("predicted_answer")
            gt        = labels_dict.get(doc_name + ".pdf", {}).get(question)
            correct   = _judge(model_mod, question, gt, predicted)
            total_tok = _count_tokens("\n".join(s.get("text", "") for s in document.get("texts", [])))
            ret_tok   = pred_rec.get("retrieved_token_count", 0)
            per_doc.append({
                "doc_name":         doc_name,
                "predicted":        predicted,
                "ground_truth":     gt,
                "correct":          correct,
                "latency_seconds":  pred_rec.get("latency_seconds", 0.0),
                "retrieved_tokens": ret_tok,
                "input_tokens":     pred_rec.get("input_tokens", 0),
                "cost_ratio":       (ret_tok / total_tok) if total_tok > 0 else 0.0,
                "used_fallback":    pred_rec.get("used_fallback"),
            })

        eval_record = {
            "question":        question,
            "question_slug":   question_slug,
            "split":           split,
            "apply_strategy":  apply_strategy,
            "n":               len(per_doc),
            "n_correct":       sum(1 for r in per_doc if r["correct"]),
            "accuracy":        round(mean(int(r["correct"]) for r in per_doc), 4) if per_doc else 0.0,
            "avg_latency":     round(mean(r["latency_seconds"] for r in per_doc), 3) if per_doc else 0.0,
            "avg_retrieved":   round(mean(r["retrieved_tokens"] for r in per_doc), 1) if per_doc else 0.0,
            "avg_input_tok":   round(mean(r["input_tokens"] for r in per_doc), 1) if per_doc else 0.0,
            "avg_cost_ratio":  round(mean(r["cost_ratio"] for r in per_doc), 6) if per_doc else 0.0,
            "per_doc":         per_doc,
        }
        _write_json(eval_out, eval_record)
        eval_results[split] = {k: v for k, v in eval_record.items() if k != "per_doc"}
        print(f"  [eval:{split}] accuracy={eval_record['accuracy']:.3f}  cost_ratio={eval_record['avg_cost_ratio']:.4f}", flush=True)

    return eval_results


def _apply_one(
    *,
    apply_strategy: str,
    document: dict,
    rule_names: list[str],
    rule_folder: Path,
    fallback_folder: Path | None,
    question: str,
    question_slug: str,
    rules_dir: Path,
    output_dir: Path,
    model: str,
):
    if apply_strategy == "merge":
        from rule_apply.merge import rule_apply_merge
        return rule_apply_merge(
            document=document, rule_names=rule_names,
            question_slug=question_slug, question=question,
            rules_dir=str(rule_folder.parent), output_dir=str(output_dir),
            model_name=model,
        )

    if apply_strategy == "default":
        if fallback_folder is None:
            raise ValueError("apply_strategy='default' requires a fallback (Stage 2 pool)")
        # `apply_with_fallback` expects a single rule_folder, but in the grid layout
        # refined rules live in <refined_root>/<q_slug>/ and the full pool lives in
        # <rules_root>/<q_slug>/ — different parents. Re-implement the gate+fallback
        # logic inline, calling rule_apply_merge with the correct rules_dir each time.
        from rule_apply.merge import rule_apply_merge
        from rule_apply.default import relevance_check, qa_call

        # Step 1: merge over the refined subset (no LLM)
        refined_res = rule_apply_merge(
            document=document, rule_names=rule_names,
            question_slug=question_slug, question=question,
            rules_dir=str(rule_folder.parent), output_dir=str(output_dir),
            model_name=model,
        )
        refined_text = refined_res.get("retrieved_text") or ""

        # Step 2: gpt54mini gate
        has_answer, rel_in, rel_out = relevance_check(refined_text, question, model_name="gpt54mini")

        # Step 3: pick retrieval source (refined vs full pool)
        used_text  = refined_text
        used_tokens = refined_res.get("retrieved_token_count", 0) or 0
        full_tokens = 0
        if not has_answer:
            full_names = _list_rule_names(fallback_folder)
            full_res = rule_apply_merge(
                document=document, rule_names=full_names,
                question_slug=question_slug, question=question,
                rules_dir=str(fallback_folder.parent), output_dir=str(output_dir),
                model_name=model,
            )
            used_text   = full_res.get("retrieved_text") or ""
            used_tokens = full_res.get("retrieved_token_count", 0) or 0
            full_tokens = used_tokens

        # Step 4: gpt54 QA
        predicted, qa_in, qa_out = qa_call(used_text, question, model_name=model)

        return {
            "doc_name":               document.get("doc_name"),
            "predicted_answer":       predicted,
            "retrieved_text":         used_text,
            "retrieved_token_count":  used_tokens,
            "retrieved_tokens_refined": refined_res.get("retrieved_token_count", 0) or 0,
            "retrieved_tokens_full":  full_tokens,
            "used_fallback":          not has_answer,
            "relevance_verdict":      "yes" if has_answer else "no",
            "input_tokens":           qa_in,
            "output_tokens":          qa_out,
            "relevance_input_tokens": rel_in,
            "relevance_output_tokens": rel_out,
        }

    raise ValueError(f"Unknown apply_strategy: {apply_strategy!r}")


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    sampling_strategy: str = "random",
    rule_gen_strategy: str = "llm_coarse",
    refine_strategy:   str = "none",
    apply_strategy:    str = "merge",
    queries_file:      str = "data/financebench/sample_queries.txt",
    dataset:           str = "financebench",
    cluster:           str = "single_cluster",
    processing_dir:    str | None = None,
    output_dir:        str = "results/e2e",
    rules_dir:         str | None = None,
    skip_existing:     bool = False,
    model:             str = "gpt54",
    stop_after:        str | None = None,
) -> dict:
    """Run the four-stage LSF pipeline once, end to end.

    Strategy choices map to:
      sampling_strategy : {"random", "fps"}
      rule_gen_strategy : {"llm_coarse", "agent_langchain", "agent_claude", "agent_codex"}
      refine_strategy   : {"none", "v1", "p_mini", "p_gpt54", "p_proxy", "p_v2", "p_v3", "agentic"}
      apply_strategy    : {"merge", "default"}

    Returns the same dict that is also persisted to {output_dir}/pipeline_summary.json.
    """
    # ── Validate ──
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}, got {sampling_strategy!r}")
    if rule_gen_strategy not in RULE_GEN_STRATEGIES:
        raise ValueError(f"rule_gen_strategy must be one of {RULE_GEN_STRATEGIES}, got {rule_gen_strategy!r}")
    if refine_strategy not in REFINE_STRATEGIES:
        raise ValueError(f"refine_strategy must be one of {REFINE_STRATEGIES}, got {refine_strategy!r}")
    if apply_strategy not in APPLY_STRATEGIES:
        raise ValueError(f"apply_strategy must be one of {APPLY_STRATEGIES}, got {apply_strategy!r}")
    if apply_strategy == "default" and refine_strategy == "none":
        raise ValueError("apply_strategy='default' needs a refined subset; pick a refine_strategy != 'none'.")

    # Staged execution: run stages up to and including `stop_after`, then stop.
    # Every stage is idempotent (skip-existing on disk), so a later invocation
    # with a further stop_after cheaply reuses everything already computed.
    _STAGES = ("sampling", "rule_gen", "precompute", "refine", "apply")
    if stop_after is not None and stop_after not in _STAGES:
        raise ValueError(f"stop_after must be one of {_STAGES}, got {stop_after!r}")
    _stop_idx = _STAGES.index(stop_after) if stop_after else len(_STAGES) - 1
    def _run(stage: str) -> bool:
        return _STAGES.index(stage) <= _stop_idx

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processing_dir = processing_dir or str(_default_processing_dir(dataset))
    # Default rules dir matches the grid layout: rules/<dataset>/grid/<sampling>/<rule_gen>/<q_slug>/
    rules_root = Path(
        rules_dir
        or (_ROOT / "rules" / dataset / "grid" / sampling_strategy / rule_gen_strategy)
    )
    # All other per-stage roots live under output_dir.
    cache_root   = out / "cache"   / sampling_strategy / rule_gen_strategy
    refined_root = out / "refined" / sampling_strategy / rule_gen_strategy / refine_strategy
    apply_root   = out / "apply"   / sampling_strategy / rule_gen_strategy / refine_strategy / apply_strategy

    print(f"\n=== Pipeline ===", flush=True)
    print(f"  sampling   : {sampling_strategy}", flush=True)
    print(f"  rule_gen   : {rule_gen_strategy}", flush=True)
    print(f"  refine     : {refine_strategy}", flush=True)
    print(f"  apply      : {apply_strategy}", flush=True)
    print(f"  dataset    : {dataset}  cluster : {cluster}", flush=True)
    print(f"  output_dir : {out}", flush=True)
    print(f"  rules_root : {rules_root}", flush=True)
    print(f"  cache_root : {cache_root}", flush=True)
    print(f"  refined    : {refined_root}", flush=True)
    print(f"  apply      : {apply_root}\n", flush=True)

    # ── Stage 1 ──
    print("Stage 1 — Sampling", flush=True)
    sample_path, unsampled_path = stage_sampling(
        strategy=sampling_strategy, dataset=dataset, cluster=cluster,
        output_dir=out, skip_existing=skip_existing,
    )
    sample_labels    = json.loads(sample_path.read_text(encoding="utf-8"))
    unsampled_labels = json.loads(unsampled_path.read_text(encoding="utf-8"))
    sample_doc_map    = _load_docs(sample_labels, processing_dir)
    unsampled_doc_map = _load_docs(unsampled_labels, processing_dir)
    sample_docs       = list(sample_doc_map.values())
    sample_doc_names  = list(sample_doc_map.keys())
    print(f"  sampled={len(sample_doc_map)}  unsampled={len(unsampled_doc_map)}", flush=True)

    questions = _load_queries(queries_file)
    print(f"  questions={len(questions)}\n", flush=True)

    if not _run("rule_gen"):
        print(f"\n=== Done (stop-after=sampling). Sampling complete for {sampling_strategy}. ===", flush=True)
        return {"stop_after": stop_after, "stage_completed": "sampling",
                "sampling_strategy": sampling_strategy,
                "num_sampled_docs": len(sample_doc_map),
                "num_unsampled_docs": len(unsampled_doc_map)}

    # ── Per-question loop: Stages 2-4 ──
    per_question_summary: list[dict] = []

    for question in questions:
        slug = _make_slug(question)
        question_slug = slug   # pure slug; sample size lives in the labels file, not the path
        print(f"\n--- Question: {question}", flush=True)

        try:
            ground_truth = {
                k.replace(".pdf", ""): v[question]
                for k, v in sample_labels.items() if question in v
            }

            # Stage 2 — Rule Generation
            print("Stage 2 — Rule Generation", flush=True)
            rule_folder_gen, gen_meta = stage_rule_gen(
                strategy=rule_gen_strategy, question=question, question_slug=question_slug,
                sample_docs=sample_docs, sample_doc_names=sample_doc_names,
                ground_truth=ground_truth, rules_dir=rules_root,
                output_dir=out, model=model, skip_existing=skip_existing,
            )
            n_rules_gen = len(_list_rule_names(rule_folder_gen))

            # Phase C — Per-pool precompute (only if the refine strategy needs it)
            base_refine, _ = _split_strategy_model(refine_strategy, default_model=model)
            if _run("precompute") and base_refine in _REFINE_NEEDS_PRECOMPUTE:
                print("Phase C — Precompute", flush=True)
                stage_precompute(
                    rule_folder=rule_folder_gen, cache_root=cache_root,
                    sample_docs=sample_doc_map, sample_labels=sample_labels,
                    question=question, question_slug=question_slug,
                    rules_dir=rules_root, skip_existing=skip_existing,
                )

            if not _run("refine"):
                continue

            # Stage 3 — Refinement (optional)
            if refine_strategy == "none":
                effective_folder = rule_folder_gen
                fallback_folder  = None
                n_rules_refined  = None
            else:
                print("Stage 3 — Rule Refinement", flush=True)
                effective_folder = stage_refine(
                    strategy=refine_strategy, question=question, question_slug=question_slug,
                    rule_folder=rule_folder_gen, sample_docs=sample_docs,
                    ground_truth=ground_truth, rules_dir=rules_root,
                    refined_root=refined_root, cache_root=cache_root,
                    model=model, skip_existing=skip_existing,
                )
                fallback_folder = rule_folder_gen if apply_strategy == "default" else None
                n_rules_refined = len(_list_rule_names(effective_folder))

            if not _run("apply"):
                continue

            # Stage 4 — Apply + Evaluate
            print("Stage 4 — Apply + Evaluate", flush=True)
            eval_results = stage_apply_and_eval(
                apply_strategy=apply_strategy, question=question, question_slug=question_slug,
                rule_folder=effective_folder, fallback_folder=fallback_folder,
                sample_docs=sample_doc_map, unsampled_docs=unsampled_doc_map,
                sample_labels=sample_labels, unsampled_labels=unsampled_labels,
                rules_dir=rules_root, apply_root=apply_root,
                model=model, skip_existing=skip_existing,
            )

            per_question_summary.append({
                "question":                 question,
                "question_slug":            question_slug,
                "num_rules_generated":      n_rules_gen,
                "num_rules_after_refine":   n_rules_refined,
                "sampled_accuracy":         eval_results.get("sampled",   {}).get("accuracy"),
                "unsampled_accuracy":       eval_results.get("unsampled", {}).get("accuracy"),
                "avg_cost_ratio_sampled":   eval_results.get("sampled",   {}).get("avg_cost_ratio"),
                "avg_cost_ratio_unsampled": eval_results.get("unsampled", {}).get("avg_cost_ratio"),
            })

        except Exception as e:
            print(f"  ERROR on {question_slug}: {e}", flush=True)
            per_question_summary.append({
                "question":      question,
                "question_slug": question_slug,
                "error":         str(e),
            })

    # Partial (staged) run — apply did not run, so there is no eval summary to write.
    if not _run("apply"):
        print(f"\n=== Done (stop-after={stop_after}). "
              f"Stages up to '{stop_after}' complete for {len(questions)} question(s). ===", flush=True)
        return {"stop_after": stop_after, "stage_completed": stop_after,
                "sampling_strategy": sampling_strategy, "rule_gen_strategy": rule_gen_strategy,
                "refine_strategy": refine_strategy, "num_questions": len(questions)}

    # ── Overall summary ──
    sampled_accs   = [q["sampled_accuracy"]   for q in per_question_summary if q.get("sampled_accuracy")   is not None]
    unsampled_accs = [q["unsampled_accuracy"] for q in per_question_summary if q.get("unsampled_accuracy") is not None]
    cost_ratios    = [q["avg_cost_ratio_unsampled"] for q in per_question_summary if q.get("avg_cost_ratio_unsampled") is not None]

    summary = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "sampling_strategy":  sampling_strategy,
        "rule_gen_strategy":  rule_gen_strategy,
        "refine_strategy":    refine_strategy,
        "apply_strategy":     apply_strategy,
        "queries_file":       queries_file,
        "dataset":            dataset,
        "cluster":            cluster,
        "num_questions":      len(questions),
        "num_sampled_docs":   len(sample_doc_map),
        "num_unsampled_docs": len(unsampled_doc_map),
        "questions":          per_question_summary,
        "overall": {
            "avg_sampled_accuracy":   round(mean(sampled_accs),   4) if sampled_accs   else None,
            "avg_unsampled_accuracy": round(mean(unsampled_accs), 4) if unsampled_accs else None,
            "avg_cost_ratio":         round(mean(cost_ratios),    6) if cost_ratios    else None,
        },
    }
    # Per-combo summary lives under apply_root so 64 combos don't clobber each other.
    summary_path = apply_root / "pipeline_summary.json"
    _write_json(summary_path, summary)
    print(f"\n=== Done. Summary -> {summary_path} ===", flush=True)
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_cli():
    p = argparse.ArgumentParser(
        description="LSF end-to-end pipeline: sampling -> rule_gen -> refine -> apply + eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sampling-strategy", default="random", choices=SAMPLING_STRATEGIES)
    p.add_argument("--rule-gen-strategy", default="llm_coarse", choices=RULE_GEN_STRATEGIES)
    p.add_argument("--refine-strategy",   default="none", choices=REFINE_STRATEGIES)
    p.add_argument("--apply-strategy",    default="merge", choices=APPLY_STRATEGIES)
    p.add_argument("--queries-file",      default="data/financebench/sample_queries.txt")
    p.add_argument("--dataset",           default="financebench")
    p.add_argument("--cluster",           default="single_cluster")
    p.add_argument("--processing-dir",    default=None)
    p.add_argument("--rules-dir",         default=None)
    p.add_argument("--output-dir",        default="results/e2e")
    p.add_argument("--model",             default="gpt54")
    p.add_argument("--skip-existing",     action="store_true")
    p.add_argument("--stop-after",        default=None,
                   choices=["sampling", "rule_gen", "precompute", "refine", "apply"],
                   help="Run stages up to and including this one, then stop. "
                        "Stages are idempotent, so later invocations reuse prior output.")
    return p


def main():
    args = _build_cli().parse_args()
    run_pipeline(
        sampling_strategy = args.sampling_strategy,
        rule_gen_strategy = args.rule_gen_strategy,
        refine_strategy   = args.refine_strategy,
        apply_strategy    = args.apply_strategy,
        queries_file      = args.queries_file,
        dataset           = args.dataset,
        cluster           = args.cluster,
        processing_dir    = args.processing_dir,
        rules_dir         = args.rules_dir,
        output_dir        = args.output_dir,
        model             = args.model,
        skip_existing     = args.skip_existing,
        stop_after        = args.stop_after,
    )


if __name__ == "__main__":
    main()
