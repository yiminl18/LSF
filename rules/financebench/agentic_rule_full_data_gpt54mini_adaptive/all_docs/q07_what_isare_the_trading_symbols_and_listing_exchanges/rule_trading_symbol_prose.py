def rule_trading_symbol_prose(doc: dict) -> list[dict]:
    try:
        import re

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        exchange_terms = (
            "new york stock exchange",
            "nyse",
            "nasdaq",
            "the nasdaq stock market",
            "the nasdaq stock market llc",
            "nasdaq global select market",
            "nasdaq global market",
            "nasdaq capital market",
            "stock exchange",
            "exchange",
        )
        action_terms = ("listed", "traded", "trades", "trade", "trading")
        symbol_terms = ("symbol", "ticker")

        sources = []
        for item in (doc.get("paragraphs") or [])[:300]:
            text = norm(item.get("text"))
            if text:
                sources.append((text, item.get("page_no"), item.get("paragraph_no"), None))
        for item in (doc.get("lines") or [])[:260]:
            text = norm(item.get("text"))
            if text:
                sources.append((text, item.get("page_no"), None, item.get("line_no")))

        out = []
        seen = set()
        for text, page_no, paragraph_no, line_no in sources:
            low = text.lower()
            if (
                "title of each class" in low
                or "securities registered pursuant to section 12" in low
                or "trading symbol" in low
                or "name of each exchange on which registered" in low
                or "name of exchange on which registered" in low
                or "commission file number" in low
                or "registrant’s telephone number" in low
                or "registrant's telephone number" in low
                or "indicate by check mark" in low
                or "table of contents" in low
            ):
                continue
            has_symbol = any(term in low for term in symbol_terms)
            has_action = any(term in low for term in action_terms)
            has_exchange = any(term in low for term in exchange_terms)
            if not (has_symbol and has_action and has_exchange):
                continue
            key = (low, page_no, paragraph_no, line_no)
            if key in seen:
                continue
            seen.add(key)
            span = {"text": text}
            if page_no is not None:
                span["page_no"] = page_no
            if paragraph_no is not None:
                span["paragraph_no"] = paragraph_no
            if line_no is not None:
                span["line_no"] = line_no
            out.append(span)
            if len(out) >= 3:
                return out

        full_text = doc.get("text") or ""
        if full_text:
            fragments = re.split(r"(?<=[.!?])\s+", full_text)
            for frag in fragments:
                text = norm(frag)
                if not text:
                    continue
                low = text.lower()
                if (
                    "title of each class" in low
                    or "securities registered pursuant to section 12" in low
                    or "trading symbol" in low
                    or "name of each exchange on which registered" in low
                    or "name of exchange on which registered" in low
                    or "commission file number" in low
                    or "registrant’s telephone number" in low
                    or "registrant's telephone number" in low
                    or "indicate by check mark" in low
                    or "table of contents" in low
                ):
                    continue
                has_symbol = any(term in low for term in symbol_terms)
                has_action = any(term in low for term in action_terms)
                has_exchange = any(term in low for term in exchange_terms)
                if not (has_symbol and has_action and has_exchange):
                    continue
                span = {"text": text}
                out.append(span)
                if len(out) >= 3:
                    break

        return out
    except Exception:
        return []
