# Improved Rule Refinement Algorithm (v2)

**Status:** Proposal
**Date:** 2026-05-16
**Replaces / extends:** `src/rule_refine.py` (cost-sorted exponential search + backward pruning)
**Motivation:** [analysis/overfit_report.md](../analysis/overfit_report.md) — avg accuracy gap of **+0.158** between sampled-refined rules and unsampled-gold rules

---

## 1. Problem Statement

The current `rule_refine.py` selects rule subsets that hit a target accuracy on 10 sampled docs but generalize poorly: average accuracy drops from **0.778** on sampled to the gold's **0.936** on unsampled (+0.158 gap). Root-cause analysis attributes the gap to three failure modes:

| # | Failure mode | Share | Symptom |
|---|--------------|------:|---------|
| 1 | Sampling blind spots | ~40% | Rules with zero sampled coverage are silently eliminated even when they generalize well |
| 2 | Small-N pruning noise | ~35% | Rules that help only 1–2 sampled docs look optional; pruned despite being critical on 10–20 unsampled docs |
| 3 | Rule specificity | ~25% | Page/date/format-specific rules match sampled docs by coincidence and add noise on unseen docs |

Beyond these, the cost-sorted search itself creates **structural exclusion**: rules with high coverage but high cost (e.g., `rule_cover_path_text_company_page1`, Acc-U=0.72, Cov-U=0.86, but cost=0.00175 vs selected rules at 0.00001–0.00011) sit at the back of the cost-sorted queue. The exponential search hits the target with cheap rules first and never evaluates them.

---

## 2. Current Algorithm — Failure Pattern Recap

```
Step 0: compute avg_cost per rule (no LLM)
Step 1: sort rules ascending by cost
Step 2: exponential search → take cheapest k=1,2,4,8,... rules until merge_acc ≥ target
Step 3: backward pruning → remove any rule in the candidate whose removal doesn't drop accuracy
```

**Concrete failures observed:**

| Question | Selected | Gold | Sampled→Unsampled drop | Root cause in this code path |
|----------|---------:|-----:|-----------------------:|------------------------------|
| Long-term debt | 6 | 11 | 0.50 → 0.46 (target itself low) | High-coverage rules (Cov-U≥0.7) excluded by cost-sort; 7 such rules never enter the candidate |
| Total assets | 27 | 14 | 0.90 → 0.78 | Backward pruning keeps redundant rules (15 only-sampled); D\*=9, so removing a noise rule never crosses the threshold |
| Shares outstanding | 7 | 9 | 0.90 → 0.66 | Cheap specific rules act as "specialists" covering 1–3 docs each; pruning can't see they fail on unseen layouts |
| Trading symbol(s) | 4 | 32 | 1.00 → 0.94 | Algorithm terminated early after 4 cheap rules hit target; never explored the 30 generalizable broader rules |

---

## 3. Design Principles for v2

The v2 algorithm is built on five principles, each tied to a specific failure mode.

| # | Principle | Targets failure |
|---|-----------|-----------------|
| **P1** | **Individual rule signals**, not only merged accuracy | Small-N pruning noise (merged signal is too coarse on 10 docs) |
| **P2** | **Cost is a constraint, not a sort key** | Structural exclusion of expensive-but-generalizable rules |
| **P3** | **Coverage and diversity** drive selection | Specialist rules that each cover 1 doc — picking them is fine, but the metric must reward breadth |
| **P4** | **Penalize specificity** detected at the source-code level | Rules with hardcoded page numbers, months, fiscal-year tokens |
| **P5** | **Stability via resampling** (k-fold over sampled docs) | Sampling blind spots — rules robust across subsets are kept; idiosyncratic ones drop |

---

## 4. Proposed Algorithm

### 4.1 High-level Flow

```
Stage A. Per-rule profiling (LLM, but cached)
   ├─ For each rule r ∈ all_rules:
   │    compute acc(r), cov(r), cost(r) on sampled docs
   │    compute spec(r) from rule source code (no LLM)
   └─ Output: profile.json
              {rule: {acc, cov, cost, spec, covered_docs: set[doc]}}

Stage B. Build the target doc set D*
   ├─ D* = {d ∈ sampled : ∃ r with r(d) correct}
   └─ Reject rules where covered_docs ∩ D* = ∅ (they never help)

Stage C. Compute composite utility u(r)
   ├─ u(r) = acc(r) × cov(r) / (cost(r) + ε) × (1 − specificity_penalty(r))
   └─ Order rules by u(r), descending

Stage D. Greedy set-cover with diversity
   ├─ S = ∅; uncovered = D*
   ├─ While uncovered ≠ ∅ AND ∃ candidate r:
   │     pick r* = argmax_{r ∉ S} | covered_docs(r) ∩ uncovered | / cost(r)
   │     S ← S ∪ {r*}
   │     uncovered ← uncovered − covered_docs(r*)
   └─ Returns minimal-cost cover of D*

Stage E. Merge accuracy verification (LLM)
   └─ acc_merge = evaluate_merge_accuracy(S)
       If acc_merge < target: add next-best rule(s) by u(r) until ≥ target or budget exhausted

Stage F. Stability filter (k-fold over sampled docs)
   ├─ Run Stages B–E on k random splits of sampled docs (k=5, with replacement)
   ├─ Score each rule by frequency of selection across folds
   └─ Final S' = {r : freq(r) ≥ ⌈k/2⌉} ∪ {r : individually pivotal in any fold}

Stage G. Cost-budget pruning (only if |S'| exceeds budget)
   └─ Remove rules whose removal doesn't drop merge_acc AND have lowest marginal coverage
```

### 4.2 Stage A — Per-rule Profiling (LLM-free)

**Key change:** Stage A uses **zero LLM calls**. We replace per-rule accuracy with a substring proxy (`proxy_judge` already in `src/rule_refinement/eval_judge.py`): a rule is treated as "correct on doc d" iff the ground-truth answer string appears in the retrieved text. Cost and coverage were always LLM-free.

```python
from rule_refinement.eval_judge import proxy_judge

def profile_rules(rules, docs, gt, rule_folder):
    """Zero-LLM profiling. Returns per-rule signals usable for ranking."""
    profile = {}
    for r in rules:
        covered_docs = set()
        proxy_correct = set()
        cost_per_doc = []
        for d in docs:
            total_toks = count_tokens(concat_texts(d.texts))
            spans = apply_rule(r, d)
            if not spans:
                cost_per_doc.append(0.0)
                continue
            covered_docs.add(d.name)
            retrieved_text = concat(spans)
            cost_per_doc.append(count_tokens(retrieved_text) / total_toks)
            if proxy_judge(gt[d.name], retrieved_text):
                proxy_correct.add(d.name)
        profile[r] = {
            "proxy_acc":    len(proxy_correct) / len(docs),
            "cov":          len(covered_docs) / len(docs),
            "cost":         mean(cost_per_doc),
            "covered_docs": covered_docs,
            "proxy_docs":   proxy_correct,
            "spec":         specificity_score(r),  # regex on source, also free
        }
    return profile
```

**Why proxy is a good ranker (but a bad final grader):**

| Proxy outcome | Reality |
|---------------|---------|
| GT string is in retrieved text | High probability the LLM will answer correctly — the answer is literally present |
| GT string absent from retrieved text | LLM might still paraphrase (e.g., "NYSE" ↔ "New York Stock Exchange", "$4.5B" ↔ "4,500 million"). The proxy under-counts in these cases. |

The under-counting is biased toward **stylistic equivalence**, not toward retrieval correctness — so it's safe for **ranking**: a rule that proxy-passes on 6 docs almost certainly retrieves more usable content than a rule that proxy-passes on 1.

**Mitigations for proxy false-negatives (numeric GTs):**

```python
def proxy_judge_fuzzy(gt, retrieved_text):
    if proxy_judge(gt, retrieved_text):
        return True
    # Numeric variants: strip $, commas, "billion"/"million", scale
    gt_variants = numeric_variants(gt)  # e.g., {"4.5", "4,500", "4500000000"}
    return any(v.lower() in retrieved_text.lower() for v in gt_variants)
```

For FinanceBench specifically, ground truths are mostly verbatim spans (addresses, phone numbers, EINs, ticker symbols) or numbers with a small set of representations — the fuzzy variant catches almost all paraphrase cases.

**LLM cost analysis (revised):**

| Stage | v1 calls | v2 calls (revised) |
|-------|---------:|-------------------:|
| Initial target_accuracy eval | 2·D | 0 (target derived from proxy union) |
| Stage A profiling | n/a | **0** |
| Stage B–D | n/a | 0 (pure Python over the profile) |
| Stage E merge verification | n/a | 2·D · (1 + iters) ≈ 20–60 |
| Stage F k-fold | n/a | k · Stage-E cost ≈ 100–300 |
| **v1 total (measured)** | **~1400** | — |
| **v2 total (revised)** | — | **~150–350** |

That is roughly **4–10× cheaper** than v1, not 50% more expensive as in the previous draft.

**Caching:** Stage A output is a pure function of (rule, doc, ground_truth). Cache as JSON keyed by rule name; rerunning after adding rules costs only the new rules.

### 4.3 Stage B — Define the Target Set D\*

```python
D_star = {d for d in sampled_docs
          if any(d in profile[r]["proxy_docs"] for r in all_rules)}
```

D\* is the **achievable ceiling**: docs where at least one rule retrieves text containing the ground-truth answer string. The algorithm's goal is to find a cheap set covering D\*.

**Key change from v1:** D\* is computed from the proxy in Stage A — zero LLM calls — instead of v1's all-rules merged evaluation that costs `2·D` calls. The proxy under-counts when answers are paraphrased; for FinanceBench this is rare because ground truths are mostly verbatim spans.

### 4.4 Stage C — Composite Utility Function

```python
def utility(r, profile, alpha=1.0, beta=1.0, eps=1e-4):
    p = profile[r]
    return (p["proxy_acc"] ** alpha) * (p["cov"] ** beta) / (p["cost"] + eps) * (1 - p["spec"])
```

**Interpretation:**
- `proxy_acc · cov` rewards rules that fire often AND retrieve the GT-bearing span.
- Dividing by `cost` keeps cheap rules competitive but doesn't exclude expensive ones — a high-proxy, high-cov rule can win even at 100× the cost.
- `(1 − spec)` damps page/date-specific rules.

**Hyperparameters:** `α=1.0, β=1.0` by default. Tune on a held-out validation question if available.

### 4.5 Stage D — Greedy Set-Cover (Diversity)

Instead of picking the top-k by utility (which often picks redundant rules covering the same docs), we use greedy set-cover. At each step we pick the rule that **adds the most uncovered D\* docs per unit cost**:

```python
def greedy_cover(profile, D_star):
    S = []
    uncovered = set(D_star)
    pool = sorted(profile, key=lambda r: -utility(r, profile))
    while uncovered and pool:
        best = max(pool,
                   key=lambda r: len(profile[r]["proxy_docs"] & uncovered)
                                  / (profile[r]["cost"] + 1e-4))
        gain = len(profile[best]["proxy_docs"] & uncovered)
        if gain == 0:
            break
        S.append(best)
        uncovered -= profile[best]["proxy_docs"]
        pool.remove(best)
    return S
```

**Why this is better than v1's pruning:**
- v1 prunes from "all cheap rules" — never includes rules outside the cheap prefix.
- v2 adds **anything that uniquely covers new docs**, regardless of cost.
- For long-term debt, the missed rules (`rule_debt_note_header_same_page_tables` etc.) have Cov-U=0.94. In Stage A, they're profiled as correct on ~5–9 sampled docs. The greedy cover picks them when they add new D\* docs even though they cost 0.005.

### 4.6 Stage E — Merge Accuracy Verification

Set-cover guarantees per-rule coverage of D\*, but the merged answer may still be wrong (rules disagree, irrelevant context confuses QA). Verify with one LLM evaluation pass:

```python
acc_merge, _ = evaluate_merge_accuracy(S, sampled_docs, ...)
while acc_merge < target_accuracy and pool:
    next_r = pool.pop(0)  # next by utility
    if next_r not in S:
        S.append(next_r)
        acc_merge, _ = evaluate_merge_accuracy(S, ...)
```

This is bounded: at worst, S grows to include all rules with non-zero D\* contribution. In practice, 1–3 extra rules suffice.

### 4.7 Specificity Detection (Stage A subroutine)

```python
PAGE_PATTERN = re.compile(r"page[_ ]?(no|number)?\s*[=<>!]+\s*\d+|page_around_\d+")
MONTH_PATTERN = re.compile(r"\b(january|february|march|april|may|june|july|august|"
                            r"september|october|november|december)\b", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"pages?_(\d+)_to_(\d+)")

def specificity_score(rule_name):
    """Returns ∈ [0, 1]. Higher = more specific."""
    src = read_rule_source(rule_name)
    score = 0.0
    if PAGE_PATTERN.search(src):    score += 0.5
    if MONTH_PATTERN.search(src):   score += 0.3
    if RANGE_PATTERN.search(src):   score += 0.4
    # Cap at 0.8 so high-specificity rules aren't fully zeroed
    return min(score, 0.8)
```

**Examples from FinanceBench:**
- `rule_page1_shares_outstanding_as_of_february` → contains "february" → spec=0.3
- `rule_tables_with_total_assets_and_page_around_39` → "page_around_39" → spec=0.5
- `rule_balance_sheet_table_on_pages_30_to_70` → "pages_30_to_70" → spec=0.4

These rules can still win if their acc·cov is high enough, but they need to beat broader alternatives.

### 4.8 Stage F — Stability via Resampling

```python
def stability_filter(all_rules, sampled_docs, k=5, frac=0.7):
    selection_freq = Counter()
    for fold in range(k):
        subset = random.sample(sampled_docs, int(len(sampled_docs) * frac))
        S_fold = run_stages_B_to_E(all_rules, subset)
        selection_freq.update(S_fold)
    threshold = math.ceil(k / 2)
    return [r for r, cnt in selection_freq.items() if cnt >= threshold]
```

**Cost:** k × (Stage A is shared; only B–E re-run). For k=5, ~5× the Stage B–E cost, but Stage A is the dominant LLM cost and is shared.

**Effect on root causes:**
- Sampling blind spots: rules robust across subsets get a vote of confidence.
- Small-N noise: idiosyncratic selections from one subset are filtered out.

### 4.9 Stage G — Final Budget Pruning (optional)

Only triggered if `|S'| > rule_budget` (e.g., 20). Removes the rule whose removal has the smallest impact on merge accuracy.

---

## 5. Walkthrough on Failure Cases

### 5.1 Long-term debt (gap +0.36)

**v1 behavior:**
- Cost-sort puts cheap shared rules first.
- Exp search at k=4 hits target 0.50 on sampled, exits.
- 7 high-coverage gold rules at cost 0.004–0.009 never evaluated.

**v2 behavior:**
- Stage A profiles all 13 rules via substring proxy → `rule_debt_note_header_same_page_tables` shows proxy_acc≈0.7+ (GT debt numbers literally appear), cov=1.00, cost=0.0058 — no LLM call needed.
- Stage C utility ≈ 0.7 × 1.0 / 0.0058 ≈ 120 — among the top rules.
- Stage D set-cover picks it because it adds 5+ new D\* docs.
- Stage E verifies merge ≥ 0.50 with the real LLM judge; likely 0.70+.

**Expected gain:** sampled→unsampled acc closer to gold's 0.82.

### 5.2 Total assets (rule bloat, 27 → 14)

**v1 behavior:**
- Backward pruning removes only rules whose removal drops acc below target.
- Many of the 15 sampled-only redundant rules each contribute marginally on different docs, none individually causes a drop, all kept.

**v2 behavior:**
- Stage D set-cover stops once D\* is covered — never adds the 15 redundant rules.
- Even if Stage F (k-fold stability) brings some back, the threshold filters out fold-specific picks.

**Expected gain:** rule count ≈ 12–16, closer to gold's 14, less retrieval noise.

### 5.3 Shares outstanding (specialist overfit)

**v1 behavior:**
- 7 cheap specialist rules each covering 1–3 sampled docs cumulatively hit 0.90.
- The expensive generalist `rule_cover_path_text_company_page1` (Acc-U=0.72) costs 100× more and never enters the cheap prefix.

**v2 behavior:**
- Stage A (substring proxy): generalist has proxy_acc≈0.7+, cov=0.86, cost=0.00175. utility ≈ 340.
- Specialists: proxy_acc=0.10–0.30, cov=0.10–0.30, cost ≈ 0.0001. utility highly variable.
- Stage D set-cover: generalist covers many D\* docs in one pick. Then 1–2 specialists fill remaining D\* gaps.
- Stage F stability: specialists like `as_of_february` only get picked when February-fiscal-year docs are in the random subset → low selection frequency → filtered out.

**Expected gain:** ~3–5 rules selected (vs 7), unsampled acc closer to gold's 0.96.

---

## 6. Computational Cost Comparison

For a question with N=100 rules, D=10 sampled docs, expected outputs S=5–10 selected rules:

| Stage | LLM calls | Notes |
|-------|----------:|-------|
| **v1 total (measured)** | **~1400** | dominated by exp search + pruning |
| v2 Stage A (profile) | **0** | substring proxy + Python (no LLM) |
| v2 Stage B (D\*) | 0 | derived from Stage A |
| v2 Stage C (utility) | 0 | scoring only |
| v2 Stage D (set-cover) | 0 | greedy over profile |
| v2 Stage E (merge verify) | 2·D · (1 + iters) ≈ 20–60 | usually 1–3 merge evals |
| v2 Stage F (k-fold) | k · Stage-E ≈ 100–300 | k=5 |
| v2 Stage G (budget) | 2·D · removed ≈ 0–40 | only if over budget |
| **v2 total** | **~150–350** | **4–10× cheaper than v1** |

**Caching wins on incremental runs:** Stage A is a pure function of rule code + doc content + ground truth — no model dependence. Adding a single new rule costs zero LLM calls; only Stage E re-runs.

**Why the cost dropped instead of growing:** the v1 algorithm runs the LLM judge inside its inner loop (every exp-search step, every prune step, every doc). The v2 algorithm runs the LLM judge only on the final merged set and during k-fold verification — the search is entirely combinatorial over a pre-computed profile.

---

## 7. Implementation Sketch

### 7.1 File layout

```
src/rule_refine_v2.py          # new entry point
src/rule_refinement/
    profile.py                 # Stage A (refactored from rule_refine.py)
    target_docs.py             # Stage B
    utility.py                 # Stage C
    set_cover.py               # Stage D
    merge_verify.py            # Stage E
    specificity.py             # Stage A subroutine
    stability.py               # Stage F
```

Most pieces can be lifted from the existing `src/rule_refinement/` modules (`select_rules.py` already implements greedy cover; `coverage_check.py` has tau-filter that's close to Stage G).

### 7.2 Public signature

```python
def rule_refine_v2(
    rule_names: list[str],
    question: str,
    question_slug: str,
    documents: list[dict],
    ground_truth: dict,
    rules_dir: str = "rules/...",
    output_dir: str = "rules/.../refine_v2",
    model_name: str = "gpt54mini",
    target_accuracy: float | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    specificity_lambda: float = 1.0,
    k_folds: int = 5,
    rule_budget: int | None = None,
    profile_cache_dir: str | None = None,
) -> dict:
    ...
```

### 7.3 Output schema (additions over v1)

```json
{
  ...                                  // all existing v1 fields
  "rule_profile": {
    "rule_name": {"acc": 0.46, "cov": 0.94, "cost": 0.0058,
                  "spec": 0.0, "covered_docs": [...], "correct_docs": [...]}
  },
  "D_star": ["doc_a", "doc_b", ...],
  "stage_traces": {
    "set_cover_picks": [...],
    "verification_iters": [...],
    "fold_selections": [[...], [...], ...]
  },
  "stability": {"rule_name": {"freq": 4, "k_folds": 5}}
}
```

The schema is a superset of v1's, so downstream consumers (eval_merge, run_merge_unsampled, the overfit report generator) continue to work.

---

## 8. Validation Plan

1. **Re-run on FinanceBench 10 questions** with the same 10 sampled docs and 50 unsampled docs.
2. **Compare** v1 vs v2 along three axes:
   - Avg accuracy gap on unsampled docs (target: < 0.05, vs v1's +0.158)
   - Rule count (target: within ±20% of gold)
   - LLM call budget (target: < 2× v1)
3. **Ablations:**
   - v2 without Stage F (no resampling) — isolate stability contribution
   - v2 without specificity penalty — isolate that lever
   - α, β grid search — find robust defaults
4. **Failure-mode checks** on the three worst v1 questions:
   - Long-term debt: does the high-coverage rule cluster get selected?
   - Total assets: does rule count drop to ≤ 16?
   - Shares outstanding: does the generalist `rule_cover_path_text_company_page1` get selected?

---

## 9. Open Questions

1. **Proxy false-negative audit:** Run Stage A with the substring proxy AND the real LLM judge on one question, measure the disagreement rate. Above some threshold (say >10%), enable the fuzzy numeric-variants proxy by default.
2. **D\* on unsampled:** If we had a small "validation" set distinct from sampled, we could measure stability on it directly. Should the algorithm reserve 2–3 of the 10 sampled docs as held-out?
3. **Adaptive k_folds:** When D\* is small (< 5), more folds help; when large, fewer suffice. k = max(3, 50 / |D\*|).
4. **Tighter specificity regex:** Current regex scans the rule source for `page_around_\d+`, month names, page-range patterns. Extend to detect numeric constants in conditional expressions (e.g., `if page_no == 39`).

---

## 10. Summary

| Lever | v1 | v2 |
|-------|----|----|
| Rule ordering | cost ascending | utility = acc · cov · spec\_dampened / cost descending |
| Inclusion rule | first k cheap rules hitting target | greedy set-cover of D\* |
| Pruning | backward, target-driven | redundancy-driven within S |
| Cost role | sort key (hard exclusion) | denominator in utility (soft penalty) |
| Stability | none | k-fold resampling vote |
| Specificity | none | regex penalty (`page_around_NN`, month names) |
| Per-rule signal | merge only | proxy_acc + cov + cost (LLM-free, cached) |
| LLM calls per question | ~1400 (measured) | ~150–350 (estimated) |
| Expected unsampled gap | +0.158 (measured) | ≤ +0.05 (target) |

The core shift is from **"find the cheapest rule set that hits target accuracy on 10 docs"** to **"find a robust set of rules whose per-doc contributions, individually validated, cover the achievable target set across resampled splits."**
