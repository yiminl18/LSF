def rule_page1_company_identity_dense_region(doc: dict) -> list[dict]:
    """Match dense page-1 identity-region spans before the first business/part section starts."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        stop_idx = None
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if re.search(r"\bPART I\b|\bITEM 1\b|TABLE OF CONTENTS|INDEX|Forward-Looking Statements|CAUTIONARY STATEMENT", txt, re.I):
                stop_idx = i
                break
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and (stop_idx is None or i < stop_idx):
                out.append(span)
        return out
    except Exception:
        return []
