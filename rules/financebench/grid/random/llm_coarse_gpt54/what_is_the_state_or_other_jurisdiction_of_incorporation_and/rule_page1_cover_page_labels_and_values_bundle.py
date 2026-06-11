def rule_page1_cover_page_labels_and_values_bundle(doc: dict) -> list[dict]:
    """Match page-1 spans in bundles where labels and values for state/EIN are adjacent."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"State or other jurisdiction|IRS Employer Identification|I\.?R\.?S\.?", txt, re.I):
                for cand in texts[max(0, i-2):min(len(texts), i+3)]:
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
