def rule_debt_limit_answer_candidates_high_recall(doc: dict) -> list[dict]:
    """High-recall rule for any span likely to answer whether the debt limit is suspended or set at a dollar amount."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"debt ceiling|debt limit|statutory limit|statutory limitation|debt subject to statutory|suspended until|automatically raised", text, re.I):
                out.append(span)
                continue
            if "federal budget and debt" in path.lower() and re.search(r"gross federal debt|debt held by the public", text, re.I):
                out.append(span)
                continue
            if "federal debt" in path.lower() and re.search(r"\b(FD|FO)[-\s]?(6|8|9)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
