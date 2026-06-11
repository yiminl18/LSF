def rule_page1_exchange_or_symbol_dense_header(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span densely contains class/symbol/exchange listing content."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            blob = (span.get("text_span", "") or "")
            score = 0
            for pat in [r"trading symbol", r"exchange", r"section 12\(b\)", r"common stock", r"nasdaq", r"stock exchange"]:
                if re.search(pat, blob, re.I):
                    score += 1
            if score >= 3:
                out.append(span)
        return out
    except Exception:
        return []
