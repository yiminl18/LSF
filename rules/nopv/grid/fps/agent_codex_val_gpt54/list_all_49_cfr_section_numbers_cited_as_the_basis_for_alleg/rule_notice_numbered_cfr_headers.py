def rule_notice_numbered_cfr_headers(doc: dict) -> list[dict]:
    """Match the first numbered section header for each cited 49 CFR section before compliance-order remedies begin."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(r"^\s*response to this notice\s*$", re.IGNORECASE)
        order_item_re = re.compile(r"\b(in regard to item|with respect to item|regarding item)\b", re.IGNORECASE)
        header_re = re.compile(
            r"^\s*\d+\s*[\.)]?\s*(?:49\s*cfr\s*)?(?:§|s)?\s*(19\d\.\d+(?:\([a-z0-9]+\))*)",
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
            if span.get("label") != "section_header":
                continue
            match = header_re.search(span.get("text") or "")
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
