def rule_receipts_answer_candidate_tables(doc: dict) -> list[dict]:
    """Broad high-recall rule for tables likely containing the fiscal-year-to-date receipts answer."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            score = 0
            if re.search(r'(Summary of Fiscal Operations|FFO[-\s]?1)', txt, re.I):
                score += 2
            if re.search(r'(Budget Receipts by Source|FFO[-\s]?2)', txt, re.I):
                score += 2
            if re.search(r'(Fiscal \d{4} to date|Actual fiscal year to date)', txt, re.I):
                score += 2
            if re.search(r'(Total receipts|Net receipts|Net budget receipts)', txt, re.I):
                score += 2
            if re.search(r'(Net outlays|Total outlays)', txt, re.I):
                score += 1
            if score >= 3:
                out.append(span)
        return out
    except Exception:
        return []
