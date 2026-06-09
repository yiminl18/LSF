def rule_page1_exact_exchange_with_registration_context(doc: dict) -> list[dict]:
    """Match short page-1 exchange-name spans that sit inside the Section 12(b) cover block."""
    try:
        import re

        exchange_re = re.compile(
            r"^(?:the )?(?:new york stock exchange|nasdaq|nasdaq global select market|nasdaq global market|nasdaq capital market|chicago stock exchange,? inc\.?)$",
            re.IGNORECASE,
        )
        inline_exchange_re = re.compile(
            r"name of (?:each )?exchange on which registered\s+(?:the )?(?:new york stock exchange|nasdaq(?: global select market| global market| capital market)?|chicago stock exchange,? inc\.?)",
            re.IGNORECASE,
        )
        anchor_re = re.compile(
            r"securities registered pursuant to section 12\(b\)|name of each exchange|trading symbol",
            re.IGNORECASE,
        )
        blocked_re = re.compile(
            r"aggregate market value|holders of record|also traded on|also listed on|closing price|number of shares",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        def is_exact_exchange(text: str) -> bool:
            normalized = normalize(text)
            if exchange_re.fullmatch(normalized):
                return True
            parts = normalized.split()
            if len(parts) > 2 and len(parts) % 2 == 0:
                mid = len(parts) // 2
                first = " ".join(parts[:mid])
                second = " ".join(parts[mid:])
                return first == second and bool(exchange_re.fullmatch(first))
            return False

        page1 = [span for span in doc.get("texts", []) if span.get("page_no") == 1 and span.get("label") != "table"]
        matched: list[dict] = []
        for idx, span in enumerate(page1):
            text = normalize(span.get("text") or "")
            is_exact = is_exact_exchange(text)
            is_inline = bool(inline_exchange_re.search(text))
            if (not is_exact and not is_inline) or blocked_re.search(text):
                continue
            window = " ".join(
                normalize(page1[j].get("text") or "") + " " + normalize(((page1[j].get("structure") or {}).get("path_text")) or "")
                for j in range(max(0, idx - 6), min(len(page1), idx + 1))
            )
            if is_inline or anchor_re.search(window):
                matched.append(span)
        return matched
    except Exception:
        return []
