def rule_8k_exhibit_table_dated_as_of(doc: dict) -> list[dict]:
    """Match exhibit tables in 8-Ks whose descriptions contain 'dated as of <date>' or 'dated <date>'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                text = (span.get("text") or "").lower()
                if "exhibit" in text and ("dated as of" in text or "dated " in text):
                    out.append(span)
        return out
    except Exception:
        return []
