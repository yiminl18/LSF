def rule_market_info_common_stock_sentence(doc: dict) -> list[dict]:
    """Match narrative sentences stating where common stock or ordinary shares trade and under what symbol."""
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
        note_count = sum(
            1 for s in texts
            if s.get("page_no") in anchor_pages
            and "notes due" in s.get("text", "").lower()
        )
        has_symbol_value = False
        for i, s in enumerate(texts):
            if s.get("page_no") not in anchor_pages:
                continue
            if "trading symbol" not in s.get("text", "").lower():
                continue
            page = s.get("page_no")
            for j in range(i + 1, min(len(texts), i + 8)):
                sj = texts[j]
                if sj.get("page_no") != page:
                    continue
                candidate = " ".join(sj.get("text", "").split())
                if re.fullmatch(r"[A-Z][A-Z0-9./%-]{0,7}", candidate):
                    has_symbol_value = True
                    break
            if has_symbol_value:
                break
        if note_count >= 3 and has_symbol_value:
            return []

        anchor_phrases = (
            "our common stock is traded",
            "our common stock has been traded",
            "our common stock is listed",
            "our common stock is currently listed",
            "the company’s common stock is traded",
            "the company's common stock is traded",
            "the company’s common stock is listed",
            "the company's common stock is listed",
            "principal market for our common stock",
            "our ordinary shares are traded",
        )
        patterns = (
            "traded on",
            "trades under",
            "listed on",
            "listed under",
            "under the symbol",
            "ticker symbol",
            "trading symbol",
            "principal market",
        )
        exchange_terms = ("nasdaq", "new york stock exchange", "nyse")
        banned_terms = (
            "as well as",
            "australian securities exchange",
            "börse stuttgart",
        )

        out = []
        for s in texts:
            text = " ".join(s.get("text", "").split())
            low = text.lower()
            path = ((s.get("structure") or {}).get("path_text") or "").lower()
            if any(b in low for b in banned_terms):
                continue
            if not (any(a in low for a in anchor_phrases) or "market for registrant" in path):
                continue
            if any(p in low for p in patterns) and any(e in low for e in exchange_terms):
                out.append(s)
        return out
    except Exception:
        return []
