def rule_nike_adjacent_share_numbers(doc: dict) -> list[dict]:
    """Retrieve split numeric cover-page spans after a registrant common-stock outstanding lead-in."""
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
            if "number of shares of the registrant's common stock outstanding" in hay or "number of shares of the registrant’s common stock outstanding" in hay:
                out.append(span)
                for j in range(i + 1, min(i + 12, n)):
                    s2 = texts[j]
                    if s2.get("label") != "text":
                        continue
                    if s2.get("page_no") != span.get("page_no"):
                        continue
                    p2 = ((s2.get("structure") or {}).get("path_text") or "")
                    if p2 != path:
                        continue
                    t2 = (s2.get("text") or "").strip()
                    if re.fullmatch(r"[\$,0-9\s]+", t2) and any(ch.isdigit() for ch in t2):
                        out.append(s2)
                return out
        return out
    except Exception:
        return []

