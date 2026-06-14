def rule_signature_line_after_sincerely_with_director(doc: dict) -> list[dict]:
    """Match the short director signature line immediately after a standalone Sincerely line."""
    try:
        import re

        texts = doc.get("texts", [])
        sincerely_re = re.compile(r"^sincerely,?$", re.I)
        director_re = re.compile(
            r"\b(?:Acting\s+)?Director\b[^\n]{0,160}\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )

        out = []
        for i, span in enumerate(texts[:-1]):
            next_span = texts[i + 1]
            text = (span.get("text") or "").strip()
            next_text = (next_span.get("text") or "").strip()
            if span.get("page_no", 0) < 2:
                continue
            if next_span.get("page_no") != span.get("page_no"):
                continue
            if sincerely_re.match(text) and director_re.search(next_text):
                out.append(next_span)
        return out
    except Exception:
        return []
