def rule_page1_address_and_zip_in_neighboring_spans(doc: dict) -> list[dict]:
    """Return page-1 neighboring spans that together form address plus ZIP around the registrant block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            neighborhood = texts[max(0, i-2): min(len(texts), i+3)]
            joined = " ".join((s.get("text") or "") for s in neighborhood)
            if re.search(r'address of principal executive offices', joined, re.I) or re.search(r'zip code', joined, re.I):
                for s in neighborhood:
                    st = (s.get("text") or "").strip()
                    if re.search(r'^\d{1,6}\s+\S+|\bone\b', st, re.I) or re.search(r'\b\d{5}(?:-\d{4})?\b', st):
                        if s not in out:
                            out.append(s)
        return out
    except Exception:
        return []
