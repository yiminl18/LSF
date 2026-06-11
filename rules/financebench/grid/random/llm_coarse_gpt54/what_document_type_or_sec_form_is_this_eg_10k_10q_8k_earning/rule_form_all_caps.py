def rule_form_all_caps(doc: dict) -> list[dict]:
    """Match all-caps-looking form spans such as FORM 10-K/10-Q/8-K."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I):
                if span.get("all_cap") == 1 or text.upper() == text:
                    out.append(span)
        return out
    except Exception:
        return []
