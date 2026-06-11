def rule_form_code_with_quarterly_report_context(doc: dict) -> list[dict]:
    """Match 10-Q form spans when QUARTERLY REPORT appears in the same or nearby cover spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if re.search(r"\bFORM\s+10-Q\b", span.get("text") or "", re.I):
                neighborhood = " ".join((texts[j].get("text") or "") for j in range(max(0, i-2), min(len(texts), i+3))).upper()
                if "QUARTERLY REPORT" in neighborhood:
                    out.append(span)
        return out
    except Exception:
        return []
