def rule_page1_exact_symbol_with_registration_context(doc: dict) -> list[dict]:
    """Match short page-1 ticker-like spans that sit inside the Section 12(b) cover block."""
    try:
        import re

        symbol_re = re.compile(r"^[A-Z0-9][A-Z0-9./-]{0,11}$")
        inline_symbol_re = re.compile(r"trading symbol(?:\(s\))?\s+([A-Z0-9][A-Z0-9./-]{0,11})\b", re.IGNORECASE)
        anchor_re = re.compile(
            r"securities registered pursuant to section 12\(b\)|trading symbol",
            re.IGNORECASE,
        )
        blocked_re = re.compile(r"^(?:FORM|YES|NO|OR|USA|NONE|COMMON|STOCK)$", re.IGNORECASE)

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        page1 = [span for span in doc.get("texts", []) if span.get("page_no") == 1 and span.get("label") != "table"]
        matched: list[dict] = []
        for idx, span in enumerate(page1):
            text = normalize(span.get("text") or "")
            is_exact = bool(symbol_re.fullmatch(text))
            is_inline = bool(inline_symbol_re.search(text))
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
