import re

def rule_cover_state_ein(doc: dict) -> list[dict]:
    """Cover-page spans labeling or containing State of Incorporation and IRS EIN, plus their preceding value spans."""
    texts = doc.get("texts", []) or []
    ein_re = re.compile(r"\b\d{2}-\d{7}\b")
    keep = set()
    for i, sp in enumerate(texts):
        page = sp.get("page_no") or 0
        if page > 3:
            continue
        t = sp.get("text") or ""
        path = ((sp.get("structure") or {}).get("path_text") or "")
        blob = (t + " " + path).lower()
        is_label = (
            "jurisdiction of incorporation" in blob
            or "state of incorporation" in blob
            or "employer identification" in blob
        )
        is_ein = bool(ein_re.search(t) or ein_re.search(path))
        if is_label or is_ein:
            keep.add(i)
            if i > 0:
                keep.add(i - 1)
    return [texts[i] for i in sorted(keep)]
