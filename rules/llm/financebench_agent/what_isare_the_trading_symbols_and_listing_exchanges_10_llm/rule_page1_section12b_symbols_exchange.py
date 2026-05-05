def rule_page1_section12b_symbols_exchange(doc: dict) -> list[dict]:
    """Retrieve page-1 Section 12(b) spans mentioning trading symbols and listing exchanges."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label")
            hay = f"{path} {txt}".lower()
            if (
                "section 12(b)" in hay
                or "trading symbol" in hay
                or "trading symbol(s)" in hay
                or "name of each exchange" in hay
                or "name of each exchange on which registered" in hay
                or "exchange on which registered" in hay
                or "nasdaq global select market" in hay
                or "new york stock exchange" in hay
                or "nyse" in hay
            ):
                if label in {"text", "section_header", "table"}:
                    out.append(span)
        return out
    except Exception:
        return []

