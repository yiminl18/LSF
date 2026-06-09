# Rule Application — Cost-Descent Prune (proposed)

> **Status: design only.** This document specifies a new rule-application strategy
> and analyzes it against the current Default (refined-with-fallback) strategy in
> [`rule_application.md`](rule_application.md) / `src/rule_apply/default.py`.
> Nothing is implemented yet — this is the write-up requested before coding.

Working names: **Cost-Descent**, **Merge + Descent**, or **Halving Prune**. (Pick one
before implementation — see [Open decisions](#open-decisions).)

---

## 1. Goal / intuition

The expensive answer model (gpt54) is billed on its **input context**. The Default
strategy feeds it the *whole refined retrieval* on the hit-path. But often the answer
lives in a small, cheap subset of the refined rules, and the rest is just padding that
costs tokens (and can distract the model).

**Cost-Descent** keeps shrinking the context — always toward the *cheapest* rules —
as long as a cheap gate model (gpt54mini) still says "the answer is in here". It then
hands the expensive model the **smallest passing subset** instead of the full refined
set. The expensive model is still called **exactly once**; we only shrink what it reads.

It degrades gracefully to Default's behavior at both ends:
- if even the full refined set doesn't contain the answer → fall back to the full
  generation pool (identical to Default's miss-path);
- if no shrinking is possible/safe → it answers from the full refined set (identical
  to Default's hit-path).

---

## 2. Inputs

| Input | Source |
|---|---|
| `refined_rules: list[str]` | refined subset — from Pareto/agentic refinement, **or** the rule set an agent returns directly in the rule-end-to-end strategy (`agentic_rule_full_data`) |
| `all_rules: list[str]` | full Step-2 generation pool (the fallback set; same as Default's `all_rules`) |
| `rule_folder: Path` | directory of `rule_<name>.py` files |
| per-rule **cost** | a scalar per rule used to order them low→high (see [Open decisions](#open-decisions)) |
| `relevance_model` | gpt54mini (cheap gate) |
| `qa_model` | gpt54 (expensive answerer) |

---

## 3. Algorithm

Sort the refined rules by cost **ascending**: `r₁ (cheapest) … rₙ (most expensive)`.
Define `top-k` = the `k` **cheapest** rules `{r₁ … r_k}`. These are **nested**:
`top-1 ⊂ top-2 ⊂ … ⊂ top-n`, so their merged retrievals are nested too — text only
grows as `k` grows. (This nesting is what makes the halving sound; see
[§5 correctness](#5-why-the-halving-is-sound).)

```
sort refined_rules by cost ascending           # r1..rn
text_prev ← retrieve(top-n)                     # = retrieve(all refined rules)

# ── Gate 0: is the answer even in the refined set? ──
if NOT relevance(text_prev, q):                 # gpt54mini
    text_used ← retrieve(all_rules)             # full generation pool
    answer    ← qa(text_used, q)                # gpt54  — DONE (fallback path)
    return  (fallback_triggered = True)

# ── Descend: halve while the gate still says YES ──
k ← n
while k // 2 >= 1:
    k_next   ← k // 2
    text_next ← retrieve(top-k_next)
    if relevance(text_next, q):                 # gpt54mini — answer still present
        text_prev ← text_next                   # accept the smaller context
        k ← k_next                              # recurse to top-k/2
    else:
        break                                   # k/2 lost the answer → keep top-k

answer ← qa(text_prev, q)                        # gpt54 on smallest passing subset — DONE
```

- The descent visits `k = n, n/2, n/4, …, 1` (integer halving), stopping at the first
  level whose gate says NO, and answers from the **last level that said YES**.
- The expensive `qa(...)` runs **once**, on `text_prev`.
- "use text from last iteration" in the spec = `text_prev` (the last accepted level).

### Counts per doc (when the refined set contains the answer)

| | gpt54mini gate calls | gpt54 answer calls |
|---|---:|---:|
| best case (top-n/2 already fails) | 2 | 1 |
| typical | `1 + ⌊log₂ n⌋` | 1 |
| n = 1 | 1 | 1 |

On the fallback path: 1 gate call + 1 answer call (same as Default's miss-path).

---

## 4. Worked example (n = 8)

Rules sorted by cost: `r1..r8`. Suppose the answer is contained in `top-2` but not
`top-1`.

| Step | Tested | Gate (mini) | Action |
|---|---|---|---|
| gate 0 | top-8 | YES | descend |
| 1 | top-4 | YES | accept, `text_prev=top-4` |
| 2 | top-2 | YES | accept, `text_prev=top-2` |
| 3 | top-1 | NO  | stop → answer from `top-2` |

Expensive model reads `top-2` (2 cheapest rules) instead of `top-8`. 4 mini gate calls,
1 gpt54 call. Default would have read all 8 rules' merged text.

If instead the answer were only in the most expensive rule `r8`: gate 0 on top-8 = YES,
top-4 (drops r8) = NO → stop immediately, answer from `top-8`. Identical cost to Default's
hit-path, plus one extra (cheap) gate call.

---

## 5. Why the halving is sound

Because `top-k` sets are nested, `retrieve(top-k)` text grows monotonically with `k`.
The predicate *"answer present in top-k"* is therefore (assuming a perfect gate)
**monotone**: there is a threshold `k*` such that all `k ≥ k*` contain the answer and
all `k < k*` don't. Descending while the gate says YES walks down toward `k*`.

Caveat — it is a **geometric** search, not exact: it returns the smallest *power-of-two*
prefix `top-(n/2ʲ)` that still passes, which can be larger than the true minimal `k*`
(e.g. if `top-2` and `top-3` pass but `top-1` fails, it stops at `top-2`, never testing
`top-3`→… it would actually land on `top-2`, which is fine; the loss is when the minimal
`k*` lies strictly between two visited powers — then we slightly over-include). This trades
a little cost-optimality for `O(log n)` gate calls instead of `O(n)`.

---

## 6. Comparison with Default

| Aspect | Default (fallback) | Cost-Descent (proposed) |
|---|---|---|
| Expensive (gpt54) calls / doc | 1 | **1** (unchanged) |
| Cheap (gpt54mini) gate calls / doc | 1 | `1 + ⌊log₂ n⌋` (hit-path) |
| Context fed to gpt54 (hit-path) | full **refined** retrieval `T_refined` | **smallest passing subset** `T_{k*} ≤ T_refined` |
| Context fed to gpt54 (miss-path) | full pool `T_full` | full pool `T_full` (identical) |
| Headline QA cost ratio (= gpt54 input / doc) | `T_refined / D` | **`T_{k*} / D ≤` Default** |
| gpt54mini gate tokens | logged, **excluded** from cost ratio | logged, **excluded** from cost ratio |
| Latency (sequential round-trips) | ≤ 2 | up to `1 + ⌊log₂ n⌋` gates **+** 1 answer |
| Main risk | one gate false-positive on refined set | **more gates ⇒ more chances for a boundary false-positive** (see below) |
| Reduces to Default when | — | `n=1`, or top-n/2 fails immediately, or refined-set gate fails (fallback) |

**Cost accounting:** the cost metric counts **gpt54 tokens only** — gpt54mini gate tokens
are treated as free and **ignored**. So the descent's extra gate calls cost *nothing* in
the headline metric; the only thing that moves the cost is the size of the single gpt54
context, `T_{k*}`. This makes Cost-Descent a **strict cost improvement** over Default on
the hit-path (`T_{k*} ≤ T_refined`, same one gpt54 call), and identical on the miss-path.

**Where it wins:** whenever the answer concentrates in a few cheap rules, the expensive
model reads far less text → lower cost, and often *higher* accuracy (less distracting
context).

**Where it can lose:**
1. **Boundary false-positive.** If the gate wrongly says YES on a subset that *doesn't*
   actually contain the answer, the expensive model answers from too little context →
   wrong / `NOT FOUND`. Default has only **one** gate (on the larger refined set, where a
   false-positive is less likely); Descent has up to `log n` gates and *acts* on the last
   YES. Mitigation: the existing relevance prompt is conservative ("if unsure, reply NO"),
   which biases toward stopping early (safe — reverts to a larger, known-good context).
2. **Latency.** Up to `log n + 1` sequential LLM round-trips vs Default's ≤ 2. (Pure
   wall-clock — the extra gpt54mini calls are free *cost-wise* but not *time-wise*.)

**Net:** Cost-Descent is a strict *cost-down* refinement of Default's hit-path that keeps
the expensive call count fixed; its accuracy is bounded above by "Default + better context
focus" and below by "Default − gate false-positives." Expected to lower the QA cost ratio
on docs where the refined set is over-broad, at the price of more cheap gate calls,
more latency, and sensitivity to gpt54mini gate errors.

---

## 7. Edge cases

| Situation | Behavior |
|---|---|
| `n = 1` | gate on top-1; YES → answer top-1; NO → fallback. (≈ Default.) |
| `n` not a power of two | integer halving `k//2`; visits `n, ⌊n/2⌋, …, 1`. |
| refined retrieval empty | gate returns NO (per `relevance_check`) → fallback path. |
| two rules retrieve identical spans | union dedupes; dropping a redundant expensive rule doesn't change text → gate still YES → free shrink. |
| gpt54mini content-filter error | treated as NO (matches Default) → conservative stop / fallback. |
| all gates YES down to top-1 | answer from top-1 (max cost saving). |

---

## 8. Reuse from the existing code

Most building blocks already exist in `src/rule_apply/default.py` and the proxy
retrieval helper — Cost-Descent is mostly orchestration on top of them:

- `_retrieve_merge(document, rule_names, rule_folder)` — pure-Python nested retrieval
  for any `top-k`. **Reuse as-is.**
- `relevance_check(text, question, model)` — the gpt54mini YES/NO gate. **Reuse as-is**
  (call it once per descent level instead of once).
- `qa_call(text, question, model)` — the single gpt54 answer call. **Reuse as-is.**
- The fallback branch (`retrieve(all_rules)` → `qa_call`) is **identical** to Default's
  miss-path. **Reuse the logic.**

New code needed: (a) cost-ordering of the refined rules, (b) the halving loop, (c) the
descent trace (which `k` levels were tested, their verdicts, the final `k*`).

**Token accounting:** the per-doc log records **both** gpt54mini (summed across all gate
levels) **and** gpt54 tokens, for a complete picture. But the reported **average cost
ratio uses gpt54 tokens only** — i.e. the single gpt54 context size (`T_{k*}` on the
hit-path, `T_full` on the fallback path) divided by doc tokens. gpt54mini is logged but
never enters the cost ratio.

Proposed entry point (mirrors `apply_with_fallback`):

```python
def apply_with_descent(
    document, question, refined_rules, all_rules, rule_folder,
    rule_cost=None,                 # dict rule_name -> cost, or None to derive per-doc
    relevance_model="gpt54mini",
    qa_model="gpt54",
) -> dict[str, Any]:
    ...
```

Returned dict extends Default's with: `descent_levels` (list of `{k, tokens, verdict}`),
`final_k`, `final_n`, and gate-token totals summed across levels.

---

## 9. Decisions (confirmed)

1. **Direction — CONFIRMED cheapest-k.** After sorting low→high cost, `top-k` = the `k`
   *cheapest* rules; the halving drops the most expensive rules first to minimize the gpt54
   context.
2. **Per-rule cost (sort key) — CONFIRMED (a) per-document retrieved-token count.** Each
   rule's cost on a given doc = the token count of what `_retrieve_merge` returns for that
   single rule on that doc. Rationale: the minimal unit is answering one question on one
   doc, so a rule's cost is exactly the tokens it retrieves on that specific doc. This is
   self-contained and works for both refined and agent rule-end-to-end rules; it costs `n`
   extra no-LLM retrieval passes per doc (cheap). Ordering is therefore **per-doc**.
3. **Name** — Cost-Descent / Merge+Descent / Halving Prune *(still to pick a final label)*.
4. **Stopping granularity** — pure halving (as specified, `O(log n)` gates), answering from
   the last YES level. No further binary search for the exact minimal `k*`.
5. **Tie-breaking** equal-cost rules — stable by rule name for determinism.

---

## 10. Quick verdict

Cost-Descent = Default's hit-path, but it spends a few extra **cheap** gate calls to hand
the **expensive** model the smallest still-sufficient context. Same expensive-call budget,
strictly-not-larger expensive context, identical fallback safety net. Expected upside:
lower headline QA cost ratio (and sometimes accuracy, via context focus). Expected
downside: more gpt54mini calls, higher latency, and accuracy exposure to gate
false-positives at the stopping boundary. Worth A/B-ing against Default on nopv/court,
reporting `final_k/n` (how much it shrank) alongside accuracy and the gpt54 cost ratio
(the only cost that counts — gpt54mini gate tokens are ignored).
