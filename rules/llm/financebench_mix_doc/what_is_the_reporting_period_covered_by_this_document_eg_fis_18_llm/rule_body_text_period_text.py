def rule_body_text_period_text(doc: dict) -> list[dict]:
    """Match body text spans whose own text is the period line."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if (span.get("structure", {}) or {}).get("level") != "Body":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r'^(for the (fiscal year|quarterly period) ended|date of report)', txt):
                out.append(span)
        return out
    except Exception:
        return []
