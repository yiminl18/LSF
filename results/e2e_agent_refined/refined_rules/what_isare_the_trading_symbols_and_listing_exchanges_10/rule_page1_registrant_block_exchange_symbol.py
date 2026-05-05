def rule_page1_registrant_block_exchange_symbol(doc: dict) -> list[dict]:
    """Retrieve page-1 registrant block spans containing both exchange and symbol cues."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = f"{path} {txt}".lower()
            if span.get("label") in {"text", "section_header"} and (
                ("trading symbol" in low and "new york stock exchange" in low)
                or ("trading symbol" in low and "nasdaq global select market" in low)
                or ("symbol" in low and "new york stock exchange" in low)
                or ("symbol" in low and "nasdaq global select market" in low)
                or ("name of each exchange on which registered" in low and "new york stock exchange" in low)
                or ("name of each exchange on which registered" in low and "nasdaq global select market" in low)
            ):
                out.append(span)
        return out
    except Exception:
        return []

