"""Stage A — LLM-free per-rule profiling.

For each rule r and each sampled doc d, compute:
  - covers(r, d)     : does the rule retrieve any spans?            (Python only)
  - cost(r, d)       : retrieved_tokens / total_doc_tokens          (tiktoken only)
  - proxy_ok(r, d)   : is ground_truth a substring of retrieved?    (substring match)
  - spec(r)          : regex-based specificity score                (regex only)

No QA calls, no judge calls. Output cached as JSON per question.
"""

from __future__ import annotations

import importlib.util
import json
import re
import warnings
from pathlib import Path
from statistics import mean

from .specificity import specificity_from_file


# ── Token counting ────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Rule loading ──────────────────────────────────────────────────────────────

def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))


# ── Proxy judge (substring + numeric variants) ────────────────────────────────

_NUM_TOKEN = re.compile(r"-?\d[\d,]*\.?\d*")
_SCALE = {
    "thousand": 1_000,
    "million":  1_000_000,
    "billion":  1_000_000_000,
    "trillion": 1_000_000_000_000,
}


def _numeric_variants(gt: str) -> set[str]:
    """For a GT like '$4.5 billion', return {'4.5', '4,500,000,000', '4500000000', ...}."""
    variants: set[str] = set()
    if not gt:
        return variants
    s = gt.lower()
    # Strip currency, parens, percent
    cleaned = s.replace("$", "").replace("usd", "").replace("(", "").replace(")", "").strip()
    matches = _NUM_TOKEN.findall(cleaned)
    for m in matches:
        bare = m.replace(",", "")
        variants.add(m)
        variants.add(bare)
        # Apply scales mentioned in the GT
        for kw, mult in _SCALE.items():
            if kw in s:
                try:
                    full = int(float(bare) * mult)
                    variants.add(f"{full:,}")
                    variants.add(str(full))
                except ValueError:
                    pass
    return {v for v in variants if v}


def proxy_judge(ground_truth, retrieved_text: str, fuzzy_numeric: bool = True) -> bool:
    """Return True iff GT (or a numeric variant) appears in retrieved text."""
    if ground_truth is None or not retrieved_text:
        return False
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    lower_retrieved = retrieved_text.lower()
    if gt_str.lower() in lower_retrieved:
        return True
    if fuzzy_numeric:
        for v in _numeric_variants(gt_str):
            if v.lower() in lower_retrieved:
                return True
    return False


# ── Stage A entry point ───────────────────────────────────────────────────────

def profile_rules(
    rule_names: list[str],
    documents: list[dict],
    ground_truth: dict,
    rule_folder: Path,
    *,
    fuzzy_numeric: bool = True,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Stage A. Profile each rule on every doc without LLM calls.

    Returns a dict keyed by rule name with fields:
      proxy_acc, cov, cost, covered_docs, proxy_docs, spec, per_doc.

    If cache_path is given and exists, load it; otherwise compute and save.
    """
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        # Convert sets back from lists
        for r, prof in cached.items():
            prof["covered_docs"] = set(prof["covered_docs"])
            prof["proxy_docs"]   = set(prof["proxy_docs"])
        return cached

    profile: dict[str, dict] = {}
    n_docs = len(documents)

    # Pre-compute total tokens per doc
    total_tokens: dict[str, int] = {}
    for doc in documents:
        doc_name = doc.get("doc_name", doc.get("origin", {}).get("filename", "unknown"))
        if doc_name.endswith(".pdf"):
            doc_name = doc_name[:-4]
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
        total_tokens[doc_name] = _count_tokens(full_text)

    for rn in rule_names:
        rule_file = rule_folder / f"{rn}.py"
        if not rule_file.exists():
            warnings.warn(f"Rule file missing, skipping: {rule_file}")
            continue
        try:
            fn = _load_rule_fn(rule_file)
        except Exception as e:
            warnings.warn(f"Cannot load {rn}: {e}")
            continue

        covered_docs: set[str] = set()
        proxy_docs:   set[str] = set()
        per_doc: list[dict] = []
        cost_ratios: list[float] = []

        for doc in documents:
            doc_name = doc.get("doc_name", doc.get("origin", {}).get("filename", "unknown"))
            if doc_name.endswith(".pdf"):
                doc_name = doc_name[:-4]

            try:
                spans = fn(doc)
            except Exception as e:
                warnings.warn(f"Rule {rn} error on {doc_name}: {e}")
                spans = []

            retrieved_text = "\n\n".join(s.get("text", "") for s in spans) if spans else ""
            retrieved_tokens = _count_tokens(retrieved_text) if retrieved_text else 0
            doc_total = total_tokens.get(doc_name, 0)
            cost_ratio = retrieved_tokens / doc_total if doc_total > 0 else 0.0
            cost_ratios.append(cost_ratio)

            fires = bool(spans)
            if fires:
                covered_docs.add(doc_name)

            gt = ground_truth.get(doc_name + ".pdf") or ground_truth.get(doc_name)
            ok = bool(fires and proxy_judge(gt, retrieved_text, fuzzy_numeric=fuzzy_numeric))
            if ok:
                proxy_docs.add(doc_name)

            per_doc.append({
                "doc_name":         doc_name,
                "fires":            fires,
                "proxy_ok":         ok,
                "retrieved_tokens": retrieved_tokens,
                "cost_ratio":       round(cost_ratio, 6),
            })

        profile[rn] = {
            "proxy_acc":    round(len(proxy_docs)   / n_docs, 4) if n_docs else 0.0,
            "cov":          round(len(covered_docs) / n_docs, 4) if n_docs else 0.0,
            "cost":         round(mean(cost_ratios), 6) if cost_ratios else 0.0,
            "covered_docs": covered_docs,
            "proxy_docs":   proxy_docs,
            "spec":         round(specificity_from_file(rule_file), 3),
            "per_doc":      per_doc,
        }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serial = {
            r: {**p,
                "covered_docs": sorted(p["covered_docs"]),
                "proxy_docs":   sorted(p["proxy_docs"])}
            for r, p in profile.items()
        }
        cache_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")

    return profile
