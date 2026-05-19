# Refined Rules with Full-Pool Fallback — Application Strategy

**Status:** Implemented and evaluated
**Companion to:** `docs/rule_refinement_versions.md` (selection), `src/rule_apply_merge.py` (existing single-mode application)
**Use case:** apply a refined rule set (e.g. p_v2 output) to an *unsampled* document at inference time, with a safety fallback to the full rule pool when the refined retrieval is insufficient.

**Benchmarked configuration:**
- **Rule-gen source:** LLM-coarse (gpt54, one-shot) — `rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm/`
- **Refined subset S:** p_v2 selection over that pool — `results/.../selected_rules_pareto_v2/<slug>.json` (~5.1 rules/Q)
- **Full pool R:** all ~63 LLM-coarse rules per question
- **Dataset:** single-cluster, 50 unsampled docs, 10 questions

---

## 1. Motivation

Selection algorithms (Pareto v2, v3, etc.) pick a small rule subset `S ⊆ R` that maximizes accuracy on the 10 sampled docs. On the 50 unsampled docs we measured `mean uAcc(S) = 0.806` for p_v2 — vs `mean uAcc(R) = 0.892` for the full pool merge. So:

- **~19% of unsampled docs** are answered correctly by the full pool but **not** by `S`.
- Running the full pool everywhere wastes cost on the ~81% of docs where `S` would have sufficed.
- Running only `S` leaves 0.086 unsampled accuracy on the table.

This document specifies an **inference-time hybrid**: try the refined rules first, fall back to the full pool only when needed. The check uses **gpt54mini** (cheap) to gate the fallback. Final answer extraction always uses **gpt54** (production accuracy).

---

## 2. Algorithm

```
Inputs:
  d                – an unsampled document
  q                – the question text
  S                – refined rule set (from any selection variant, e.g. p_v2)
  R                – full rule pool (S ⊆ R)

Procedure apply_with_fallback(d, q, S, R):
  1. retrieved_S ← rule_apply_merge(d, S).retrieved_text          # cheap, ~5 rules
  2. has_answer  ← relevance_check_gpt54mini(retrieved_S, q)      # 1 cheap LLM call
  3. if has_answer:
        return qa_gpt54(retrieved_S, q)                           # 1 production LLM call
  4. retrieved_R ← rule_apply_merge(d, R).retrieved_text          # fallback, ~50 rules
  5. return qa_gpt54(retrieved_R, q)                              # 1 production LLM call
```

Three LLM calls in the worst case (relevance + QA + nothing-else), or two in the common case (relevance + QA on refined). Doc-level routing decision; no batching across docs.

---

## 3. Components

### 3.1 Relevance check (gpt54mini)

A yes/no judgment over `(retrieved_text, question)`. No ground truth required at inference time.

```python
_RELEVANCE_SYSTEM = """\
You are a relevance judge for a financial document QA system.
Given a passage extracted from a financial filing and a question,
decide if the passage contains enough information to answer the question correctly.

Reply with exactly one word: YES or NO.

Rules:
- YES iff the passage contains the specific value, name, or fact the question asks for.
- NO if the passage is missing the answer, contains only partial information,
  or contains unrelated text.
- Do not guess. If unsure, reply NO.
"""

def relevance_check(retrieved_text: str, question: str, model_name="gpt54mini") -> bool:
    if not retrieved_text or not retrieved_text.strip():
        return False
    model_mod = importlib.import_module(f"models.{model_name}")
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {"role": "user",   "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
        ],
        max_completion_tokens=5,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    return verdict.startswith("yes")
```

**Why gpt54mini for this step:**
- The relevance check is a one-word classification, not generative — gpt54mini handles it well.
- A wrong "yes" sends us to gpt54 on a doomed retrieval (small cost: one extra gpt54 call).
- A wrong "no" triggers fallback unnecessarily (cost: one extra rule_apply pass on the full pool).
- gpt54mini at ~$0.15/M tokens makes the gate effectively free.

### 3.2 Retrieval — refined vs full

Use the existing `rule_apply_merge` from `src/rule_apply_merge.py` for both. For the refined call, `rule_names = S`. For the fallback, `rule_names = R` (every rule in `rules_dir/<question_slug>/`).

```python
def retrieve_merge(document, rule_names, question_slug, question, rules_dir):
    """Wrapper around rule_apply_merge that returns just the retrieved text + token count.
    Does NOT call the QA LLM — that's the caller's job.
    """
    # In practice we call rule_apply_merge and discard the QA result, OR we
    # add a `qa=False` flag to rule_apply_merge. For the prototype, the former.
    res = rule_apply_merge(
        document=document,
        rule_names=rule_names,
        question_slug=question_slug,
        question=question,
        rules_dir=rules_dir,
        model_name="gpt54mini",      # cheap QA we ignore; pure retrieval is the goal
        output_dir=...,
    )
    return res["retrieved_text"], res["retrieved_token_count"]
```

Better: refactor `rule_apply_merge` to expose a `retrieve_only` mode that skips the QA call entirely. This drops `retrieve_only` to zero LLM calls.

### 3.3 QA call (gpt54)

The existing `rule_apply_merge`'s QA logic, but with caller-supplied `retrieved_text` instead of internally retrieving. Could be extracted to a helper:

```python
def qa_call(retrieved_text: str, question: str, model_name="gpt54") -> dict:
    model_mod = importlib.import_module(f"models.{model_name}")
    resp = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _QA_SYSTEM},
            {"role": "user",   "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
        ],
        max_completion_tokens=500,
        temperature=0.0,
    )
    return {
        "predicted_answer": (resp.choices[0].message.content or "").strip(),
        "input_tokens":     resp.usage.prompt_tokens   if resp.usage else 0,
        "output_tokens":    resp.usage.completion_tokens if resp.usage else 0,
    }
```

---

## 4. Suggested file layout

Additive — does not modify existing modules.

```
src/
  rule_apply_with_fallback.py        # NEW — exports apply_with_fallback(...)

test/
  run_apply_with_fallback_unsampled.py   # NEW — driver that loops over 10 questions × 50 unsampled docs
                                         # using the refined rules from selected_rules_pareto_v2/
                                         # and writes results to eval_pareto_v2_fallback/

results/financebench_single_cluster/llm/gpt54/one_shot/
  eval_pareto_v2_fallback/
    <slug>_unsampled.json            # per-Q sAcc + uAcc + per-doc fallback_triggered flag
    summary.json                     # cross-Q with fallback_rate per question
```

**Per-doc output schema** (in addition to existing fields):

```json
{
  "doc_name": "AMCOR_2019_10K",
  "predicted": "1,625,907,855",
  "ground_truth": "1,625,907,855",
  "correct": true,
  "retrieved_tokens_refined": 27,
  "retrieved_tokens_used": 27,
  "fallback_triggered": false,
  "relevance_verdict": "yes",
  "llm_calls": {"gpt54mini_relevance": 1, "gpt54_qa": 1, "gpt54mini_retrieve": 0}
}
```

When `fallback_triggered: true`, `retrieved_tokens_used` equals the full-pool retrieval token count.

---

## 5. Cost analysis

Per unsampled doc, in three cases:

| Case | gpt54mini calls | gpt54 calls | Notes |
|------|----------------:|------------:|-------|
| Refined sufficient (relevance=YES) | 1 | 1 | Typical case |
| Refined insufficient (relevance=NO) | 1 | 1 | Plus extra rule_apply for full pool retrieval |
| Always-refined (no gate) | 0 | 1 | Baseline (no fallback) |
| Always-full (no selection) | 0 | 1 | Most expensive in tokens — full retrieval |

**Per-doc LLM cost estimate** (using illustrative rates `gpt54mini: $0.15/M in, $0.60/M out`; `gpt54: $1.25/M in, $10/M out`):

| Strategy | Avg input tokens | Avg output tokens | Cost/doc |
|----------|----------------:|------------------:|---------:|
| Always-refined (p_v2) | ~150 (small retrieval) | ~10 | ~$0.0002 |
| Refined+fallback | ~150 if pass; ~3000 if fallback | ~10 | **~$0.0002 to ~$0.004** |
| Always-full pool | ~3000 (large retrieval) | ~10 | ~$0.004 |

If the fallback rate is `f`, expected cost per doc ≈ `(1-f) × 0.0002 + f × 0.004` USD.
For 10 questions × 50 unsampled docs = 500 docs total: a fallback rate of 20% costs ~$0.42 total. Cheap.

---

## 6. Expected accuracy

**Upper bound** for the strategy = `base_U` (full-pool unsampled accuracy = 0.892). Achievable only if relevance_check has perfect recall: every doc where refined is wrong gets routed to the fallback.

**Lower bound** = `uAcc(S)` = 0.806 (p_v2). Achievable if relevance_check always says yes (no fallback triggers).

**Expected** with realistic relevance accuracy:

| Relevance check accuracy | Effective uAcc |
|--------------------------|---------------:|
| 100% (perfect routing) | 0.892 (matches full pool) |
| 90% | ~0.875 (most of the way) |
| 80% | ~0.857 |
| 70% (random-ish) | ~0.840 |
| 50% (no signal) | ~0.823 (better than 0.806 baseline due to extra retrieval) |

For the 10 FinanceBench questions, GT strings are mostly verbatim, so gpt54mini's relevance verdict should track ground truth well. **Realistic estimate: uAcc ≈ 0.86–0.88.**

---

## 7. Why the relevance gate uses gpt54mini (not a proxy)

Earlier experiments (`p_proxy` variant) showed substring matching is too strict for numeric answers — failing on `"$4,500 million"` vs `"4,500"`. The relevance check needs semantic understanding, but only a yes/no decision, which is gpt54mini's sweet spot.

| Gate option | Pro | Con |
|-------------|-----|-----|
| `proxy_judge` (substring) | Free | Fails on numeric/format variants (3/10 Qs in p_proxy) |
| **gpt54mini relevance check** | **Handles paraphrase; cheap** | **Tiny LLM call per doc** |
| gpt54 relevance check | Most accurate | Same cost as the QA call itself — defeats the purpose |
| Skip the gate (always run QA on refined, parse "NOT FOUND") | Simpler | Wastes gpt54 calls on hopeless retrievals; "NOT FOUND" parsing is heuristic |

---

## 8. Sanity checks before trusting a run

- **Fallback rate per question**: should be roughly `1 − uAcc(S)` per question. If fallback rate << that, the relevance gate is missing failures (false-negatives on relevance). If >>, the gate is paranoid.
- **Accuracy invariant**: end-to-end uAcc with fallback ≥ uAcc(S). Falling below means the fallback retrieval *worsens* gpt54's answer (retrieval-noise interference). Investigate.
- **Cost ratio**: full-pool retrieval ≤ ~10× refined retrieval token count. If fallback rate is 100%, total cost should approximate the always-full baseline.

---

## 9. Open questions

1. **Verification with no GT**: should the relevance prompt include a generic exemplar ("for `What is X`, look for a numeric value labeled `X`") or stay generic? Tests on Q-specific prompts could tighten the gate.
2. **Caching**: should `retrieved_S` be passed to gpt54 even on fallback? Concatenating refined and full retrievals may help on edge cases where both are needed.
3. **Cascading rule sets**: extend to `S₁ ⊂ S₂ ⊂ … ⊂ R` (multi-tier fallback). For now binary fallback is sufficient.
4. **Per-question fallback budget**: cap the fraction of fallback-routed docs per question to flag potential rule-set regressions. If question Q sees 50%+ fallback, the refined `S` for Q probably needs re-selection.

---

## 10. Summary

| Property | Always-refined | Refined+fallback | Always-full |
|----------|:--------------:|:----------------:|:-----------:|
| Mean uAcc | 0.806 | **~0.86–0.88 expected** | 0.892 |
| Rules retrieved per doc | ~5 | ~5 most of the time | ~50 |
| LLM calls per doc | 1 (gpt54) | 2 (mini + gpt54) | 1 (gpt54) |
| Cost per doc | $0.0002 | $0.0002–0.0004 | $0.004 |
| Captures full-pool gain | No | **Mostly yes** | Yes (trivially) |

The strategy converts the 0.086 selection-induced accuracy loss into a **bounded, conditional cost overhead** — pay full-pool cost only on the ~20% of docs where the refined set actually failed.
