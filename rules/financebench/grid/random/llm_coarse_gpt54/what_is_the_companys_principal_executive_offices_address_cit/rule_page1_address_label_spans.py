def rule_page1_address_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans explicitly labeled as principal executive offices/address."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r'principal executive offices', txt, re.I) or re.search(r'address of principal executive offices', txt, re.I):
                out.append(span)
                if i > 0:
                    out.append(texts[i - 1])
                if i + 1 < len(texts):
                    out.append(texts[i + 1])
        return out
    except Exception:
        return []
