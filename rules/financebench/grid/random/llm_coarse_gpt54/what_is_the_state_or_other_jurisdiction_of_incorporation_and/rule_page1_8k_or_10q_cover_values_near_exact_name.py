def rule_page1_8k_or_10q_cover_values_near_exact_name(doc: dict) -> list[dict]:
    """Match page-1 spans near the exact-name caption that are likely state/EIN values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and "Exact name of registrant as specified in its charter" in txt:
                for cand in texts[i:i+20]:
                    ctext = (cand.get("text", "") or "").strip()
                    if re.fullmatch(r"\d{2}-\d{7}", ctext) or re.fullmatch(r"Delaware|Washington|New York|Jersey(?: \(Channel Islands\))?", ctext, re.I):
                        out.append(cand)
                break
        return out
    except Exception:
        return []
