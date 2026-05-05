def rule_cover_page_numeric_share_spans(doc: dict) -> list[dict]:
    """Retrieve page 1-2 cover-page body spans tied to common-stock outstanding disclosures, including adjacent numeric-only spans."""
    try:
        import re
        texts = doc.get("texts", []) or []
        out = []
        n = len(texts)
        for i, span in enumerate(texts):
            if span.get("label") != "text":
                continue
            if span.get("page_no") not in (1, 2):
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            hay = f"{path} {txt}".lower()
            is_numericish = bool(re.fullmatch(r"[\$\s,0-9\.]+", txt.strip())) and any(ch.isdigit() for ch in txt)
            direct = (
                "issued and outstanding" in hay
                or "shares of our common stock" in hay
                or "shares of common stock" in hay
                or "common stock outstanding" in hay
                or "number of shares of common stock outstanding" in hay
                or "number of shares outstanding" in hay
            )
            nearby = False
            if is_numericish:
                for j in (i-3, i-2, i-1, i+1, i+2, i+3):
                    if 0 <= j < n:
                        s2 = texts[j]
                        if s2.get("label") != "text":
                            continue
                        if s2.get("page_no") != span.get("page_no"):
                            continue
                        p2 = ((s2.get("structure") or {}).get("path_text") or "")
                        t2 = (s2.get("text") or "")
                        h2 = f"{p2} {t2}".lower()
                        if (
                            "common stock outstanding" in h2
                            or "shares of common stock" in h2
                            or "shares of our common stock" in h2
                            or "issued and outstanding" in h2
                            or "number of shares" in h2
                        ):
                            nearby = True
                            break
            if direct or nearby:
                out.append(span)
        return out
    except Exception:
        return []

