def rule_page1_exchange_after_symbol(doc: dict) -> list[dict]:
    """Match page-1 spans immediately following symbol-like spans when they look like exchanges."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        sym = re.compile(r"^[A-Z]{1,6}(?:/[0-9]{2})?$")
        for i, span in enumerate(texts[:-1]):
            txt = (span.get("text") or "").strip()
            nxt = texts[i + 1]
            nxt_txt = (nxt.get("text") or "").lower()
            if span.get("page_no") == 1 and nxt.get("page_no") == 1 and sym.match(txt):
                if "exchange" in nxt_txt or "nasdaq" in nxt_txt:
                    out.append(nxt)
        return out
    except Exception:
        return []
