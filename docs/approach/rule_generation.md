# Rule Generation

This document describes every rule generation approach in the LSF codebase. Each approach takes a question and a set of sampled documents and produces Python span-retrieval rules of the form `def rule_<name>(doc: dict) -> list[dict]`. These rules are evaluated downstream by `rule_apply_merge` (span union → LLM QA) or consumed by selection algorithms.

**Evaluation dataset:** FinanceBench. `sAcc` = merge accuracy on 10 sampled docs, `uAcc` = merge accuracy on 50 unsampled docs, `cost` = mean(retrieved\_tokens / total\_doc\_tokens).

---

## Results summary

| Method | Code | Cluster | Model | sAcc | uAcc | cost\_s | cost\_u |
|--------|------|---------|-------|-----:|-----:|--------:|--------:|
| LLM-Coarse | `src/rule_gen/llm_coarse.py` | single | gpt54 | 0.910 | **0.892** | 0.172 | 0.169 |
| Agent-Raw (LangChain) | `src/rule_gen/agent_langchain.py` | single | gpt54 | 0.860 | 0.754 | 0.116 | 0.106 |
| Agent-Raw (LangChain) | `src/rule_gen/agent_langchain.py` | single | opus47 | 0.889 | 0.776 | 0.131 | 0.120 |
| Agent-Refined (LangChain) | post-process on Agent-Raw | single | gpt54 | 0.880 | 0.734 | **0.036** | **0.036** |
| Agent-Raw (LangChain) | `src/rule_gen/agent_langchain.py` | multi | gpt54 | 0.856 | 0.843 | 0.068 | 0.087 |
| Agent-Raw (LangChain) | `src/rule_gen/agent_langchain.py` | multi | opus47 | 0.852 | 0.705 | 0.013 | 0.013 |
| Agentic-gen (random) | `agent/run_agent_gen.py` | single | opus47 | **0.960** | 0.820 | 0.006 | 0.005 |
| Agentic-gen (FPS) | `agent/run_agent_gen.py` | single | opus47 | **0.960** | 0.786 | 0.008 | 0.008 |
| Agentic-gen (Codex) | `src/rule_gen/agent_codex.py` | single | gpt54 | — | — | — | — |

> "single cluster" = 10 sampled / 50 unsampled, all 10-K. "multi cluster" = 18 sampled / 96 unsampled, mixed doc types (10-K, 10-Q, 8-K, earnings).

**Key takeaways:**
- Agentic-gen achieves the highest sAcc (0.960) and lowest cost (~37× cheaper than LLM-coarse on unsampled), but has the largest generalization gap (sAcc − uAcc = 0.14).
- LLM-coarse has the best uAcc (0.892) and smallest generalization gap (+0.018) — broad rules generalize better.
- Multi-cluster sampling closes the generalization gap dramatically (uAcc 0.843 for agent-raw gpt54 vs 0.754 single-cluster).
- Agent-Refined lowers cost 3× but regresses uAcc — downstream Pareto/agentic selection handles this trade-off better.

---

## Approach 1 — LLM-Coarse ⭐ Recommended

**Code:** `src/rule_gen/llm_coarse.py`  
**Doc:** `docs/rule_gen_llm_coarse.md`

### Description

Single LLM call with all sampled documents and ground-truth answers. The LLM is instructed to generate as many Python span-retrieval rules as possible covering every observable pattern across the sample. No iterative feedback — rules are emitted once and saved.

This is the baseline rule pool that all other approaches are compared against. It produces ~100 rules per question and achieves the highest unsampled accuracy of any single-cluster method, at the cost of high retrieval width (~17% of doc tokens per query).

### Interface

```python
def rule_gen_llm_coarse(
    documents: list[dict],     # loaded *_reconstructed.json dicts
    question: str,
    ground_truth: dict,        # { "DOCNAME.pdf": "answer string" }
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen",
    rules_dir: str = "rules/financebench",
) -> dict
```

Rules output: `def rule_<name>(doc: dict) -> list[dict]` — span objects from `doc["texts"]`.

### Design

- **Prompt structure:** question + ground truth + first 80 spans of each doc as JSON, with 7 hint categories (physical location, semantic location, keyword proximity, data feature, typography, structural position, any other).
- **Goal:** high recall. Individual rules may be imprecise; a downstream merge step union-retrieves from all rules before calling the LLM.
- **No verification at generation time.** Accuracy is only measured post-hoc by `run_eval_merge_sampled.py`.

### Results (FinanceBench, single cluster)

| Model | sAcc | uAcc | cost\_s | cost\_u | Rules/question |
|-------|-----:|-----:|--------:|--------:|---------------:|
| gpt54 | 0.910 | 0.892 | 0.172 | 0.169 | ~100 |

**Output:** `rules/financebench/lsf/single_cluster/llm/gpt54/one_shot/<slug>_10_llm/`

---

## Approach 2 — Agent-Raw / LangChain

**Code:** `src/rule_gen/agent_langchain.py`  
**Doc:** `docs/rule_gen_agent.md`

### Description

A LangChain `AgentExecutor` agent (Azure LLM backend) that iteratively generates, tests, and refines span-retrieval rules against the sampled documents. The agent uses a structured tool set with fast substring-match feedback, reserving LLM judge calls only for final union verification.

Key improvements over a single-shot approach:
- **Coverage-first rule design**: broad rules first, targeted rules for remaining gaps.
- **Fast feedback loop**: `test_rule` uses substring match (no LLM), so the agent can iterate cheaply. LLM judge only called in `test_union`.
- **Diagnosis-driven refinement**: `diagnose_failing_doc` shows a structural diff between failing and passing docs, giving the agent targeted evidence to fix specific failures.
- **Cost reduction phase**: after hitting accuracy target, agent tightens high-cost rules without dropping accuracy.

### Interface

```python
def rule_gen_agent(
    documents: list[dict],
    question: str,
    ground_truth: dict,        # { "DOCNAME.pdf": "answer string" }
    model_name: str = "gpt54",
    rules_dir: str = "rules/llm/financebench",
    output_dir: str = "results/financebench/rule_gen",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 12,
) -> dict
```

### Tool set

| Tool | LLM? | Purpose |
|------|------|---------|
| `summarize_answer_locations()` | No | Cross-doc pattern table: where does each doc's answer appear structurally (mandatory first call) |
| `inspect_answer_context(doc_name)` | No | 5 spans before/after the answer span in a specific doc |
| `write_rule(rule_name, code)` | No | Validate + register a candidate rule (rejects bad code at tool level) |
| `test_rule(rule_name)` | No | Per-doc hit rate + cost ratio via substring match |
| `show_uncovered_docs()` | No | Which docs no current rule hits |
| `diagnose_failing_doc(doc_name)` | No | Structural diff between failing doc and passing docs |
| `test_union()` | Yes | True merge accuracy (LLM QA + judge) — called sparingly |

### Agent workflow

```
summarize_answer_locations()          # mandatory first
→ write_rule() + test_rule()          # fast, no LLM
→ show_uncovered_docs()
→ diagnose_failing_doc()              # for each uncovered doc
→ write_rule() + test_rule()          # targeted fix
→ test_union()                        # first LLM call, only when covered
→ cost reduction phase (tighten high-cost rules, verify with test_union)
```

Primary objective: `merge_accuracy >= 0.90`. Secondary: minimize `avg_cost_ratio`.

### Results (FinanceBench)

| Cluster | Model | sAcc | uAcc | cost\_s | cost\_u |
|---------|-------|-----:|-----:|--------:|--------:|
| single | gpt54 | 0.860 | 0.754 | 0.116 | 0.106 |
| single | opus47 | 0.889 | 0.776 | 0.131 | 0.120 |
| single, refined | gpt54 | 0.880 | 0.734 | 0.036 | 0.036 |
| multi | gpt54 | 0.856 | **0.843** | 0.068 | 0.087 |
| multi | opus47 | 0.852 | 0.705 | **0.013** | **0.013** |

**Output:**
- `rules/financebench/lsf/single_cluster/agent/{gpt54,opus47}/raw/<slug>_10_agent/`
- `rules/financebench/lsf/single_cluster/agent/gpt54/refined/<slug>_10_agent_refined/`
- `rules/financebench/lsf/multi_clusters/agent/{gpt54,opus47}/raw/<slug>_18_agent/`

---

## Approach 3 — Agentic-gen (Claude Code) ⭐ Recommended

**Code:** `agent/run_agent_gen.py` (driver) + `src/rule_gen/agent_claude.py` (prompt builder)  
**Doc:** `docs/rule_generation_agentic_from_pdf.md`

### Description

Claude Opus 4.7 (`claude -p`) acts as a free-form agent that generates a minimal rule set from scratch. The agent starts with no rules at all — only the question, ground-truth labels, and reconstructed JSON for each sampled doc. It inspects JSON spans, identifies where answers live, and writes Python rule functions, iterating until a hard accuracy constraint is met.

Unlike the LangChain approach (Approach 2), this agent has unrestricted tool access (file read/write, shell, Python exec) and uses the same five verification tools as `rule_selection_agentic`. The generation loop is structurally identical to selection — only the inspection block differs (`read_doc_json` + `write_rule` instead of pool browsing).

The driver (`run_agent_gen.py`) spawns one `claude --dangerously-skip-permissions -p <prompt>` subprocess per question and captures the JSONL trace. The prompt builder (`rule_gen_agent_claude.py`) assembles the full task prompt with all constraints and tool descriptions.

### Constraints

| Type | Target | Verification tool |
|------|--------|-------------------|
| Hard | `match_rate = 1.0` on every sampled doc | `verify_accuracy --d-star-mode all_labeled` |
| Soft | Minimize `avg_cost_ratio` | `compute_cost` |
| Soft | Keep `|R|` in range 5–10 (fewer overfits, more generalizes better) | Agent working memory |

Budget: 30 `verify_accuracy` calls per question. Typical output: 5–10 rules per question (1–2 rules tends to overfit to the sampled docs and generalize poorly).

### Interface (prompt builder)

```python
# src/rule_gen/agent_claude.py
def build_prompt(
    question: str,
    doc_list: list[str],       # sampled doc stems
    labels_file: str,
    processing_dir: str,
    rules_dir: str,
    model: str = "opus",       # "opus" | "sonnet" | "haiku"
) -> str

def run(
    question: str,
    doc_list: list[str],
    labels_file: str,
    processing_dir: str,
    rules_dir: str,
    model: str = "opus",
    cwd: str = ".",
) -> str                       # raw claude output
```

### Results (FinanceBench, single cluster, opus47)

| Sample set | sAcc | uAcc | cost\_s | cost\_u | Rules/question |
|------------|-----:|-----:|--------:|--------:|---------------:|
| random (Task 1) | **0.960** | 0.820 | 0.006 | **0.005** | ~5–10 |
| FPS (Task 2) | **0.960** | 0.786 | 0.008 | 0.008 | ~5–10 |

Highest sAcc of all single-cluster methods. Cost is ~37× cheaper than LLM-coarse on unsampled (0.005 vs 0.169). FPS sampling did not improve unsampled generalization over random — the random 10-doc sample was already diverse enough for these questions.

**Output:**
- `rules/financebench/lsf/single_cluster/agent/opus47/agentic/raw/<slug>_10_agentic/` (Task 1)
- `rules/financebench/lsf/single_cluster/agent/opus47/agentic_fps/raw/<slug>_10_agentic_fps/` (Task 2)
- Agent traces: `results/.../agent_trace/<slug>.jsonl`

---

## Approach 3b — Agentic-gen (Codex)

**Code:** `src/rule_gen/agent_codex.py`

### Description

Direct Codex equivalent of Approach 3. Uses exactly the same task prompt, hints,
constraints, and content — only the underlying CLI is swapped from `claude -p`
to `codex exec`. There is **no Claude Code outer wrapper**: this driver spawns a
single Codex agent session per question, and that session performs all rule
generation work end to end (inspect docs, write rules, iterate).

The intent is to isolate the effect of the agent backbone (Claude Opus vs. Codex
gpt54/gpt54mini) while keeping the prompt, tool semantics, and objectives
identical. Anything else that differs between Approach 3 and Approach 3b is a
property of the underlying CLI, not the prompt.

### Models

- `gpt54` — `gpt-5.4` via Codex CLI
- `gpt54mini` — `gpt-5.4-mini` via Codex CLI

### Interface

```python
def build_prompt(
    question: str,
    docs: list[str],                    # DOC_NAMEs (no .pdf suffix)
    labels_file: str = "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
    processing_dir: str = "data/financebench/processing",
    rules_dir: str = "rules/financebench/lsf/single_cluster/agent/gpt54/codex/raw",
    model: str = "gpt54",
) -> str

def run(
    question: str,
    docs: list[str],
    labels_file: str = ...,
    processing_dir: str = ...,
    rules_dir: str = ...,
    model: str = "gpt54",
    cwd: str | None = None,
    timeout: int = 5400,
    output_last_message: str | None = None,
) -> str
```

### Usage

```bash
# Print the prompt without invoking codex (for inspection)
python src/rule_gen/agent_codex.py \
    "What is the registrant's telephone number?" \
    --docs AMCOR_2019_10K BOEING_2018_10K \
    --print-prompt

# Actually run codex on the question
python src/rule_gen/agent_codex.py \
    "What is the registrant's telephone number?" \
    --docs AMCOR_2019_10K BOEING_2018_10K \
    --model gpt54
```

### Status

Implemented. No benchmark results yet.

---

## End-to-End Note

`Agentic Rule Full Data (Codex)` was moved to
[rule_end_to_end.md](/Users/yiminglin/Documents/Codebase/LSF/docs/approach/rule_end_to_end.md)
because it is best understood as an end-to-end strategy:

- rule generation over the full corpus
- followed by downstream rule application
- followed by final QA evaluation

It is no longer listed here as a pure rule-generation approach.

---

## Approach 4 — Agent-Coarse (spec only)

**Code:** `src/rule_gen/agent_coarse.py` — **not yet implemented**  
**Doc:** `docs/rule_gen_agent_coarse.md`

### Description

LangChain `AgentExecutor` agent with a `test_rules` tool that evaluates candidate rules using **LLM-judged QA** (not substring match). The agent proposes rule code, calls `test_rules` to get per-doc LLM accuracy, analyzes failures, and refines until union coverage is maximized or `max_iterations` is reached.

Compared to Approach 2 (LangChain), the key difference is that feedback at each iteration uses the full LLM judge rather than substring match — more accurate but more expensive per iteration.

### Interface (spec)

```python
def rule_gen_agent_coarse(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen_agent_coarse",
    rules_dir: str = "rules/financebench",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 10,
) -> dict
```

Rules output: `def rule_<name>(doc: dict) -> list[dict]` — same interface as LLM-coarse and Approach 2.

### Status

Spec complete. No results — code not implemented.

---

## Approach 5 — Agent-Exact (spec only)

**Code:** `src/rule_gen/agent_exact.py` — **not yet implemented**  
**Doc:** `docs/rule_gen_agent_exact.md`

### Description

LangChain agent that generates rules whose output is the **exact answer string** rather than a list of spans. No downstream LLM call is needed — the rule output is the final answer. The agent iterates with exact-string match against ground truth.

| Aspect | LLM-Coarse / Agent-Raw | Agent-Exact |
|--------|------------------------|-------------|
| Rule output | `list[dict]` spans | `str` exact answer |
| Downstream LLM | Yes (spans → LLM → answer) | No |
| Verification | Merge accuracy (LLM judge) | Exact match |

### Interface (spec)

```python
def rule_gen_agent_exact(
    documents: list[dict],
    question: str,
    ground_truth: dict,
    model_name: str = "gpt54",
    output_dir: str = "results/financebench/rule_gen_exact",
    rules_dir: str = "rules_exact/financebench",
    logs_dir: str = "logs/financebench/agent",
    max_iterations: int = 10,
) -> dict
```

Rules output: `def rule_<name>(doc: dict) -> str` — exact answer string or `""` if not found.

**Output dir:** `rules_exact/financebench/<slug>_<n>/` (separate from span-retrieval rules).

### Status

Spec complete. No results — code not implemented.

---

## Approach comparison

| Approach | Code exists | Rule output | Feedback loop | LLM calls/iter | Best sAcc | Best uAcc |
|----------|-------------|-------------|---------------|---------------:|----------:|----------:|
| LLM-Coarse | Yes | spans | None (single shot) | 1 total | 0.910 | 0.892 |
| Agent-Raw LangChain | Yes | spans | Substring match + LLM union check | ~3–5 | 0.889 | 0.843 |
| Agentic-gen Claude | Yes | spans | LLM verify (hard constraint) | per verify call | 0.960 | 0.820 | 5–10 rules recommended |
| Agentic-gen Codex | Yes | spans | LLM verify (hard constraint) | per verify call | — | — | Same prompt as Agentic-gen Claude; CLI swapped to `codex exec` |
| Agent-Coarse | Spec only | spans | LLM per-rule QA | O(rules × iters) | — | — |
| Agent-Exact | Spec only | exact string | Exact match | 0 | — | — |

---

## Document structure for rules

All approaches write rules to a common folder layout consumable by `rule_apply_merge`:

```
rules/
└── <dataset>_<cluster>/
    ├── llm/<model>/<variant>/<slug>_<n>_llm/
    │   └── rule_<name>.py
    └── agent/<model>/<variant>/<slug>_<n>_agent/
        └── rule_<name>.py
```

Each rule file exports exactly one function:

```python
def rule_<name>(doc: dict) -> list[dict]:
    """One-line description."""
    return [span for span in doc["texts"] if ...]
```

Rules must never raise exceptions and must be self-contained (imports inside the function body).
