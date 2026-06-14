def rule_notice_direct_citation_lines(doc: dict) -> list[dict]:
    """Match short notice lines that directly name a child basis section or a cited through-range."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(r"^\s*response to this notice\s*$", re.IGNORECASE)
        order_item_re = re.compile(r"\b(in regard to item|with respect to item|regarding item)\b", re.IGNORECASE)
        direct_re = re.compile(r"^\s*§\s*(19\d\.\d+(?:\([a-z0-9]+\))*)", re.IGNORECASE)
        through_re = re.compile(
            r"in accordance with\s+§\s*19\d\.\d+[a-z0-9().-]*\s+through\s+§\s*19\d\.\d+[a-z0-9().-]*",
            re.IGNORECASE,
        )

        cutoff = len(texts)
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            if stop_re.match(text):
                cutoff = i
                break
            if span.get("label") in {"text", "list_item"} and order_item_re.search(text):
                cutoff = i
                break

        out = []
        seen = set()
        for span in texts[:cutoff]:
            if span.get("label") not in {"text", "list_item", "section_header"}:
                continue
            text = span.get("text") or ""
            if order_item_re.search(text):
                continue
            direct_match = direct_re.search(text)
            if not direct_match and not through_re.search(text):
                continue
            key = (direct_match.group(1) if direct_match else " ".join(text.split())).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(span)
        return out
    except Exception:
        return []
