def rule_federal_budget_heading_window(doc: dict) -> list[dict]:
    try:
        spans: list[dict] = []
        seen = set()
        paragraphs = doc.get("paragraphs") or []

        heading_terms = (
            "federal budget deficit and debt",
            "federal budget deficit",
            "federal budget",
            "federal deficit",
            "federal outlays and receipts as a share of gross national product",
            "budget results",
        )

        percent_terms = (
            "percent of gdp",
            "percent of gnp",
            "% of gdp",
            "% of gnp",
            "share of gdp",
            "share of gnp",
            "share of nominal gdp",
            "gross national product",
        )

        def compact(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        for i, para in enumerate(paragraphs):
            text = (para.get("text") or "").strip()
            low = text.lower()
            ctext = compact(text)
            if not text:
                continue
            if not any(term in low or compact(term) in ctext for term in heading_terms):
                continue

            window_end = min(len(paragraphs), i + 3)
            for j in range(i, window_end):
                cur = paragraphs[j]
                cur_text = (cur.get("text") or "").strip()
                cur_low = cur_text.lower()
                cur_compact = compact(cur_text)
                if not cur_text:
                    continue
                if not any(term in cur_low or compact(term) in cur_compact for term in percent_terms) and not any(
                    term in cur_low or term in cur_compact
                    for term in ("deficit", "surplus", "outlays", "receipts")
                ):
                    continue
                key = (
                    cur.get("page_no"),
                    cur.get("paragraph_no"),
                    cur_text,
                )
                if key in seen:
                    continue
                seen.add(key)
                spans.append({k: v for k, v in cur.items() if k in {"page_no", "paragraph_no", "text"}})

        return spans
    except Exception:
        return []
