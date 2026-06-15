def rule_page1_2_short_symbol_exchange_values(doc: dict) -> list[dict]:
    """Match short page 1-2 Section 12(b) symbol and exchange value spans when no table is present."""
    try:
        import re

        texts = doc.get("texts", [])
        has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
            for s in texts
        )
        if has_table:
            return []

        anchor_pages = {
            s.get("page_no")
            for s in texts
            if s.get("page_no", 999) <= 2
            and "securities registered pursuant to section 12(b)"
            in s.get("text", "").lower()
        }
        if not anchor_pages:
            return []

        out = []
        for s in texts:
            if s.get("page_no") not in anchor_pages:
                continue
            text = " ".join(s.get("text", "").split())
            low = text.lower()
            if not text or len(text) > 80:
                continue
            if any(h in low for h in ("trading symbol", "name of each exchange", "title of each class")):
                out.append(s)
                continue
            if any(h in low for h in ("nasdaq", "new york stock exchange", "nyse", "exchange llc", "global select market")):
                out.append(s)
                continue
            if re.fullmatch(r"[A-Z0-9./%-]{1,12}", text):
                out.append(s)
                continue
        return out
    except Exception:
        return []
