def rule_cover_date_in_h1_or_text_early(doc: dict) -> list[dict]:
    """Match early-page H1/text spans that are likely the issue date regardless of exact formatting."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"march|june|september|december|january|february|april|may|july|august|october|november|"
            r"spring|summer|fall|winter|fiscal"
            r")\b",
            re.I,
        )
        out = []
        for s in texts:
            if s.get("page_no", 999) <= 10 and s.get("label") in {"text", "section_header", "page_header"}:
                if pat.search((s.get("text") or "").strip()):
                    out.append(s)
        return out
    except Exception:
        return []
