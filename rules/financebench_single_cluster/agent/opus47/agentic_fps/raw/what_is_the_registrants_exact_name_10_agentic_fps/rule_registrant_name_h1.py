import re


def rule_registrant_name_h1(doc: dict) -> list[dict]:
    """Registrant H1 on page 1: the H1 immediately preceding the '(Exact name of registrant)' disclaimer, or the largest page-1 H1 when that disclaimer is absent (Edgar-style cover)."""
    texts = doc.get("texts", [])
    pat = re.compile(r"exact name of\s+(the\s+)?registrant", re.IGNORECASE)
    # Anchor-based path: H1 directly above the SEC disclaimer.
    for i, span in enumerate(texts):
        if span.get("page_no") != 1:
            continue
        if not pat.search(span.get("text", "") or ""):
            continue
        for j in range(i - 1, -1, -1):
            cand = texts[j]
            if cand.get("page_no") != 1:
                break
            struct = cand.get("structure") or {}
            if struct.get("level") == "H1" and cand.get("label") == "section_header":
                return [cand]
    # Fallback for Edgar-style covers: largest H1 section_header on page 1.
    candidates = [
        t for t in texts
        if t.get("page_no") == 1
        and t.get("label") == "section_header"
        and (t.get("structure") or {}).get("level") == "H1"
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda t: t.get("size", 0) or 0)]
