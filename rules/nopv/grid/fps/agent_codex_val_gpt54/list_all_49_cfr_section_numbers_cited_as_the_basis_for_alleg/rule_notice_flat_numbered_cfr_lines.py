def rule_notice_flat_numbered_cfr_lines(doc: dict) -> list[dict]:
    """Match flat text or list-item notice lines whose numbered item starts with a cited 49 CFR section."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(r"^\s*response to this notice\s*$", re.IGNORECASE)
        order_item_re = re.compile(r"\b(in regard to item|with respect to item|regarding item)\b", re.IGNORECASE)
        numbered_re = re.compile(
            r"^\s*\d+\s*[\.)]?\s*(?:49\s*cfr\s*)?(?:§|s)?\s*(19\d\.\d+(?:\([a-z0-9]+\))*)",
            re.IGNORECASE,
        )
        ocr_re = re.compile(r"^\s*s\s*(19\d\.\d+(?:\([a-z0-9]+\))*)", re.IGNORECASE)

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
            if span.get("label") not in {"text", "list_item"}:
                continue
            text = span.get("text") or ""
            match = numbered_re.search(text) or ocr_re.search(text)
            if not match:
                continue
            citation = match.group(1).lower()
            if citation in seen:
                continue
            seen.add(citation)
            out.append(span)
        return out
    except Exception:
        return []
