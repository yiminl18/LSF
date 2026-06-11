def rule_text_value_followed_by_label_span(doc: dict) -> list[dict]:
    """Match value spans on page 1 immediately followed by a label span for incorporation or IRS number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if span.get("page_no") != 1 or nxt.get("page_no") != 1:
                continue
            val = (span.get("text") or "").strip()
            nxt_txt = (nxt.get("text") or "").strip()
            if re.fullmatch(r"\d{2}-\d{7}", val) and re.search(r"employer\s+identification", nxt_txt, re.I):
                out.append(span)
                out.append(nxt)
            if re.search(r"^(delaware|new york|new jersey|washington|california|minnesota|jersey)$", val, re.I) and re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", nxt_txt, re.I):
                out.append(span)
                out.append(nxt)
        return out
    except Exception:
        return []
