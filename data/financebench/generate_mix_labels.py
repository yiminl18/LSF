"""
Generate mix_doc_labels.json: 25 each of 10K, 10Q, 8K, EARNINGS docs.
Selection criterion: for sample_queries.txt, most docs should have answers.
Uses existing format_labels.json where possible; calls Claude API for the rest.
"""

import json
import os
import time
import random
import anthropic

BASE = "/Users/yiminglin/Documents/Codebase/LSF/data/financebench"
TEXT_DIR = f"{BASE}/text"
NOT_FOUND_TERMS = {
    "none", "not released", "not found", "not disclosed",
    "n/a", "not applicable", "not stated", "not available",
}

def load_existing():
    with open(f"{BASE}/format_labels.json") as f:
        return json.load(f)

def load_queries():
    with open(f"{BASE}/queries.txt") as f:
        return [l.strip() for l in f if l.strip()]

def load_sample_queries():
    with open(f"{BASE}/sample_queries.txt") as f:
        return [l.strip() for l in f if l.strip()]

def get_all_docs_by_type():
    text_files = set(
        f.replace(".txt", ".pdf")
        for f in os.listdir(TEXT_DIR) if f.endswith(".txt")
    )
    cats = {"10K": [], "10Q": [], "8K": [], "EARNINGS": []}
    for f in sorted(text_files):
        for t in ["10Q", "8K", "EARNINGS", "10K"]:
            if t.upper() in f.upper():
                cats[t].append(f)
                break
    return cats

def is_not_found(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in NOT_FOUND_TERMS:
        return True
    return False

def answer_score(doc_labels, sample_queries):
    """Returns number of sample_queries answered (not-found)."""
    answered = 0
    for q in sample_queries:
        if q in doc_labels and not is_not_found(doc_labels[q]):
            answered += 1
    return answered

def generate_labels_for_doc(client, doc_name, queries, doc_text):
    """Call Claude API once per doc with all queries batched."""
    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    prompt = (
        f"Answer each numbered question about the document below. "
        f"For each answer, respond on its own line as: [N] <answer>. "
        f"If information is not present in the document, respond with: [N] None\n\n"
        f"Questions:\n{questions_block}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": (
                "You are a financial document analyst. "
                "Answer questions concisely and accurately based solely on the document provided. "
                "If the answer is a list, provide all items comma-separated. "
                "If information is not present, respond with exactly: None\n\n"
                f"DOCUMENT:\n{doc_text}"
            ),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    raw = next(b.text for b in response.content if b.type == "text")

    # Parse [N] answer lines
    result = {}
    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            try:
                idx = int(line[1:bracket_end]) - 1
                answer_text = line[bracket_end+1:].strip()
                if idx < len(queries):
                    val = None if answer_text.lower() == "none" else answer_text
                    result[queries[idx]] = val
            except ValueError:
                pass

    # Fill any missing queries with None
    for q in queries:
        if q not in result:
            result[q] = None

    return result, response.usage

def main():
    random.seed(42)
    client = anthropic.Anthropic()
    existing = load_existing()
    all_queries = load_queries()
    sample_queries = load_sample_queries()
    cats = get_all_docs_by_type()

    print(f"Loaded {len(existing)} existing labels")
    print(f"Available: " + ", ".join(f"{t}:{len(v)}" for t, v in cats.items()))

    # Separate labeled vs unlabeled per type
    labeled_by_type = {}
    unlabeled_by_type = {}
    for t, docs in cats.items():
        labeled_by_type[t] = [d for d in docs if d in existing]
        unlabeled_by_type[t] = [d for d in docs if d not in existing]

    # Determine docs to generate (need 25 per type; use labeled first)
    NEED = 25
    to_generate = {}
    to_use_existing = {}
    for t in ["10K", "10Q", "8K", "EARNINGS"]:
        labeled = labeled_by_type[t]
        unlabeled = unlabeled_by_type[t]
        if len(labeled) >= NEED:
            # Have enough labeled; score and pick top 25
            scored = sorted(
                labeled,
                key=lambda d: answer_score(existing[d], sample_queries),
                reverse=True
            )
            to_use_existing[t] = scored[:NEED]
            to_generate[t] = []
        else:
            to_use_existing[t] = labeled  # use all labeled
            needed = NEED - len(labeled)
            to_generate[t] = unlabeled[:needed]  # generate for remainder

    for t in ["10K", "10Q", "8K", "EARNINGS"]:
        print(f"\n{t}: using {len(to_use_existing[t])} existing, "
              f"generating {len(to_generate[t])} new")

    # Generate labels for unlabeled docs
    new_labels = {}
    all_to_gen = []
    for t in ["10K", "10Q", "8K", "EARNINGS"]:
        for doc in to_generate[t]:
            all_to_gen.append((t, doc))

    print(f"\nGenerating labels for {len(all_to_gen)} docs...")
    total_in, total_out = 0, 0

    for i, (doc_type, doc_name) in enumerate(all_to_gen, 1):
        txt_path = os.path.join(TEXT_DIR, doc_name.replace(".pdf", ".txt"))
        if not os.path.exists(txt_path):
            print(f"  [{i}/{len(all_to_gen)}] SKIP (no text): {doc_name}")
            new_labels[doc_name] = {q: None for q in all_queries}
            continue

        with open(txt_path, encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()

        print(f"  [{i}/{len(all_to_gen)}] {doc_type}: {doc_name} ({len(doc_text):,} chars)")
        start = time.time()
        try:
            labels, usage = generate_labels_for_doc(client, doc_name, all_queries, doc_text)
            elapsed = round(time.time() - start, 1)
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            answered = answer_score(labels, sample_queries)
            print(f"    Done in {elapsed}s | {usage.input_tokens}in/{usage.output_tokens}out | "
                  f"{answered}/{len(sample_queries)} sample queries answered")
            new_labels[doc_name] = labels
        except Exception as e:
            print(f"    ERROR: {e}")
            new_labels[doc_name] = {q: None for q in all_queries}

        time.sleep(0.5)

    print(f"\nTotal tokens: {total_in:,} in, {total_out:,} out")

    # Build final mix (25 per type), re-scoring after generation
    all_available = {}
    for t in ["10K", "10Q", "8K", "EARNINGS"]:
        pool = {}
        for doc in to_use_existing[t]:
            pool[doc] = existing[doc]
        for doc in to_generate[t]:
            if doc in new_labels:
                pool[doc] = new_labels[doc]
        # Score and pick top 25
        scored = sorted(
            pool.items(),
            key=lambda x: answer_score(x[1], sample_queries),
            reverse=True
        )
        all_available[t] = scored[:NEED]
        print(f"\n{t} selected {len(all_available[t])} docs:")
        for doc, lbl in all_available[t]:
            score = answer_score(lbl, sample_queries)
            print(f"  {doc}: {score}/{len(sample_queries)} sample queries answered")

    # Build mix_doc_labels.json
    mix = {}
    for t, docs in all_available.items():
        for doc, labels in docs:
            mix[doc] = labels

    out_path = f"{BASE}/mix_doc_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mix, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(mix)} docs to {out_path}")

    # Report answer-not-found rates per type
    print("\n=== Answer-Not-Found Rate by Doc Type (sample_queries) ===")
    for t, docs in all_available.items():
        total_q = 0
        not_found_q = 0
        for doc, labels in docs:
            for q in sample_queries:
                total_q += 1
                if is_not_found(labels.get(q)):
                    not_found_q += 1
        pct = not_found_q / total_q * 100 if total_q else 0
        print(f"  {t:12s}: {pct:5.1f}% not-found  ({not_found_q}/{total_q})")

if __name__ == "__main__":
    main()
