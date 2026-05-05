"""
Iteratively generate 20 queries answerable on >95% of the 100 sampled docs
across all 4 doc types (10K, 10Q, 8K, EARNINGS) using the OpenAI API.

Steps:
  1. Propose ~35 candidate queries based on cross-type document structure.
  2. Label all 100 sampled docs for every candidate query (GPT, batched per doc).
  3. Filter to queries with >=95% answer rate.
  4. If <20 pass, generate more candidates and repeat (up to 3 rounds).
  5. Save new_queries.json in the same format as mix_doc_labels.json.
"""

import json, os, sys, time

BASE = "/Users/yiminglin/Documents/Codebase/LSF/data/financebench"
TEXT_DIR = f"{BASE}/text"
CACHE_FILE = f"{BASE}/stats/new_query_labels_cache.json"

sys.path.insert(0, os.path.join(os.path.dirname(BASE), "..", "src"))
from models.gpt54 import chat_completions
THRESHOLD = 95.0  # minimum answer rate %
TARGET = 20
NOT_FOUND = {
    "none", "not found", "not stated", "not disclosed", "not available",
    "n/a", "not applicable", "not released", "unknown", "not provided",
    "not mentioned", "not specified", "not reported", "not present",
}

# ── Candidate queries (round 1): chosen to match structure common to all 4 types ──
ROUND1_CANDIDATES = [
    "What is the exact name of the company (registrant) this document relates to?",
    "What document type or SEC form is this (e.g. 10-K, 10-Q, 8-K, earnings release, annual report)?",
    "What is the date of this filing or the date this document was issued/published?",
    "What is the reporting period covered by this document (e.g. fiscal year ended, quarter ended, or event date)?",
    "What is the company's primary stock ticker symbol as stated in this document?",
    "What stock exchange is the company's primary common stock listed on?",
    "What is net income (or net earnings) for the most recently reported period in this document?",
    "What is total revenue (or net sales) for the most recently reported period in this document?",
    "What is diluted earnings per share (EPS) for the most recently reported period in this document?",
    "What is operating income (or operating earnings/loss) for the most recently reported period in this document?",
    "What is the company's state or jurisdiction of incorporation as stated in this document?",
    "What is the company's registrant telephone number as stated in this document?",
    "What is the company's principal executive offices address (city and state/country) as stated in this document?",
    "What is the commission file number or CIK assigned to this registrant?",
    "What is the number of shares of common stock outstanding as stated in this document?",
    "What is interest expense for the most recently reported period in this document?",
    "What is income before income taxes for the most recently reported period in this document?",
    "What is the provision for income taxes for the most recently reported period in this document?",
    "What is total assets as reported in this document?",
    "What is total liabilities as reported in this document?",
    "What is cash and cash equivalents at the end of the period as reported in this document?",
    "What is the par value per share of the company's common stock as stated in this document?",
    "Is the registrant classified as a large accelerated filer, accelerated filer, non-accelerated filer, smaller reporting company, or emerging growth company?",
    "What is the name of the CEO or principal executive officer mentioned in this document?",
    "Does this document contain or refer to forward-looking statements (Yes/No)?",
    "What is gross profit (or gross margin) for the most recently reported period in this document?",
    "What is selling, general and administrative (SG&A) expense for the most recently reported period?",
    "What is the IRS Employer Identification Number (EIN) of the registrant?",
    "What is the fiscal year-end month or quarter-end date for the period reported?",
    "What is net income per basic share for the most recently reported period?",
    "What is total stockholders' equity (or shareholders' equity) as reported in this document?",
    "What is the company's primary industry or business description in one sentence?",
    "What is the name of the stock exchange commission file number listed on the cover page?",
    "Does this document include a consolidated balance sheet or statement of financial position (Yes/No)?",
    "Does this document include a consolidated income statement or statement of operations (Yes/No)?",
]

ROUND2_CANDIDATES = [
    "What is depreciation and amortization expense for the most recently reported period?",
    "What is capital expenditure (capex) for the most recently reported period?",
    "What is free cash flow as reported or implied in this document?",
    "What is the dividend per share declared or paid as stated in this document (0 or None if not stated)?",
    "What is EBITDA or adjusted EBITDA for the most recently reported period as stated in this document?",
    "What is total debt (short-term plus long-term) as reported in this document?",
    "What is the number of full-time employees as stated in this document?",
    "What is net cash provided by operating activities for the most recently reported period?",
    "What is the company's website or investor relations URL as stated in this document?",
    "What is the aggregate market value of shares held by non-affiliates as stated in this document?",
]

ROUND3_CANDIDATES = [
    "What is the date of this document (report date, filing date, or publication date)?",
    "Is this registrant an emerging growth company as indicated in this document (Yes/No)?",
    "Does this document include a signature or authorization by a named company officer (Yes/No)?",
    "Does this document mention the name of the company's Chief Executive Officer (CEO) or equivalent?",
    "Does this document discuss or mention risks, uncertainties, or risk factors (Yes/No)?",
    "Does this document reference a stock ticker symbol or securities exchange (Yes/No)?",
    "Does this document reference or relate to the U.S. Securities and Exchange Commission (SEC) (Yes/No)?",
    "What is the company's ZIP code or postal code as stated in this document?",
    "Does this document contain any earnings per share (EPS) figures (Yes/No)?",
    "What currency is used for the financial figures in this document (e.g. USD, EUR)?",
    "Does this document mention any share repurchases or buybacks (Yes/No)?",
    "Does this document contain a table of contents or index of sections (Yes/No)?",
    "What fiscal quarter or fiscal year does the primary financial data in this document cover?",
    "Does this document mention the company's total number of locations, stores, or facilities?",
    "Does this document include a contact name or investor relations contact (Yes/No)?",
    "What is the par value per share of the company's common stock as stated in this document (None if not stated)?",
    "Is the company identified as a shell company in this document (Yes/No)?",
]


def is_not_found(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in NOT_FOUND:
        return True
    return False


def build_prompt(queries, doc_text):
    q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    return (
        "You are a financial document analyst. Answer each numbered question based solely "
        "on the document below. Be concise. If the information is not present in the document, "
        "respond with exactly: None\n\n"
        f"Questions:\n{q_block}\n\n"
        "Respond as:\n[1] <answer>\n[2] <answer>\n...\n\n"
        f"DOCUMENT:\n{doc_text[:120000]}"
    )


def parse_responses(raw, queries):
    result = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("[") and "]" in line:
            try:
                end = line.index("]")
                idx = int(line[1:end]) - 1
                answer = line[end+1:].strip()
                if 0 <= idx < len(queries):
                    result[queries[idx]] = None if answer.lower() == "none" else answer
            except ValueError:
                pass
    for q in queries:
        if q not in result:
            result[q] = None
    return result


def label_docs(client, docs_to_label, queries, cache):
    """Label all docs for the given queries, using cache where possible."""
    for doc_name in docs_to_label:
        if doc_name not in cache:
            cache[doc_name] = {}
        needed = [q for q in queries if q not in cache[doc_name]]
        if not needed:
            continue

        txt_path = os.path.join(TEXT_DIR, doc_name.replace(".pdf", ".txt"))
        if not os.path.exists(txt_path):
            for q in needed:
                cache[doc_name][q] = None
            continue

        with open(txt_path, encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()

        prompt = build_prompt(needed, doc_text)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            parsed = parse_responses(raw, needed)
            cache[doc_name].update(parsed)
        except Exception as e:
            print(f"    ERROR {doc_name}: {e}")
            for q in needed:
                cache[doc_name][q] = None
        time.sleep(0.2)
    return cache


def answer_rate(docs, query, cache):
    answered = sum(1 for d in docs if not is_not_found(cache.get(d, {}).get(query)))
    return answered / len(docs) * 100


def main():
    with open(f"{BASE}/mix_doc_labels.json") as f:
        mix = json.load(f)
    docs = list(mix.keys())
    print(f"Loaded {len(docs)} sampled docs")

    # Load or init cache
    os.makedirs(f"{BASE}/stats", exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} docs")
    else:
        cache = {}

    passing_queries = {}  # query -> answer_rate

    ROUND4_CANDIDATES = [
        "Does this document mention any named executive officer or company officer by name (Yes/No)?",
        "Does this document include any footnotes or notes to financial statements (Yes/No)?",
        "Does this document mention any legal proceedings, litigation, or regulatory matters (Yes/No)?",
        "Does this document include any non-GAAP or adjusted financial measures (Yes/No)?",
        "Does this document mention any acquisitions, divestitures, or strategic transactions (Yes/No)?",
        "Does this document include any dividend information or shareholder return information (Yes/No)?",
        "Does this document reference any fiscal year or fiscal quarter by name or number (Yes/No)?",
        "Does this document mention any macroeconomic conditions, inflation, or interest rate environment (Yes/No)?",
        "Does this document mention the company's competitive position or market share (Yes/No)?",
        "Does this document include any guidance or outlook for future periods (Yes/No)?",
    ]

    all_rounds = [
        ("Round 1", ROUND1_CANDIDATES),
        ("Round 2", ROUND2_CANDIDATES),
        ("Round 3", ROUND3_CANDIDATES),
        ("Round 4", ROUND4_CANDIDATES),
    ]

    for round_name, candidates in all_rounds:
        if len(passing_queries) >= TARGET:
            break

        new_candidates = [q for q in candidates if q not in passing_queries]
        print(f"\n{'='*60}")
        print(f"{round_name}: labeling {len(docs)} docs × {len(new_candidates)} queries")

        total = len(docs)
        for i, doc in enumerate(docs, 1):
            needed = [q for q in new_candidates if q not in cache.get(doc, {})]
            if not needed:
                continue
            txt_path = os.path.join(TEXT_DIR, doc.replace(".pdf", ".txt"))
            if not os.path.exists(txt_path):
                if doc not in cache:
                    cache[doc] = {}
                for q in needed:
                    cache[doc][q] = None
                continue

            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                doc_text = f.read()

            prompt = build_prompt(needed, doc_text)
            try:
                raw = chat_completions(prompt, max_completion_tokens=2048)
                parsed = parse_responses(raw, needed)
                if doc not in cache:
                    cache[doc] = {}
                cache[doc].update(parsed)
            except Exception as e:
                print(f"  ERROR {doc}: {e}")
                if doc not in cache:
                    cache[doc] = {}
                for q in needed:
                    cache[doc][q] = None

            if i % 10 == 0:
                print(f"  [{i}/{total}] labeled...")
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
            time.sleep(0.15)

        # Save cache after each round
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        # Evaluate
        print(f"\n{round_name} results:")
        for q in new_candidates:
            rate = answer_rate(docs, q, cache)
            print(f"  {rate:5.1f}%  {q[:90]}")
            if rate >= THRESHOLD:
                passing_queries[q] = rate

        print(f"\nPassing queries so far: {len(passing_queries)}/{TARGET}")

    # If still under target, report what we have
    final_queries = sorted(passing_queries.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print(f"Final passing queries ({len(final_queries)} with >={THRESHOLD}%):")
    for q, r in final_queries:
        print(f"  {r:5.1f}%  {q}")

    # Take top TARGET queries (or all if fewer)
    selected = [q for q, _ in final_queries[:TARGET]]

    # Build new_queries.json: same format as mix_doc_labels.json
    new_labels = {}
    for doc in docs:
        new_labels[doc] = {q: cache.get(doc, {}).get(q) for q in selected}

    out_path = f"{BASE}/new_queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(new_labels)} docs × {len(selected)} queries to {out_path}")

    # Save query list
    with open(f"{BASE}/new_queries.txt", "w") as f:
        f.write("\n".join(selected) + "\n")
    print(f"Saved query list to new_queries.txt")

    # Stats
    print("\n=== Answer-Not-Found Rate by Doc Type ===")
    type_map = {"10K": [], "10Q": [], "8K": [], "EARNINGS": []}
    for doc in docs:
        for t in ["10Q", "8K", "EARNINGS", "10K"]:
            if t.upper() in doc.upper():
                type_map[t].append(doc)
                break
    for t, tdocs in type_map.items():
        total_q = len(tdocs) * len(selected)
        nf = sum(1 for d in tdocs for q in selected if is_not_found(new_labels[d].get(q)))
        print(f"  {t:12s}: {nf/total_q*100:5.1f}% not-found ({nf}/{total_q})")


if __name__ == "__main__":
    main()
