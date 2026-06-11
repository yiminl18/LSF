def rule_balance_sheet_debt_row_generic(doc: dict) -> list[dict]:
    """Match balance sheet tables with debt rows when long-term debt may be labeled simply as 'Debt' in long-term liabilities."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            path = (span.get("structure") or {}).get("path_text", "") or ""
            if not re.search(r"balance sheet", text + " " + path, re.I):
                continue
            if re.search(r"long-term liabilities", text, re.I) and re.search(r"\n\|\s*Debt\s*\|", text, re.I):
                out.append(span)
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "") or ""))
            has_long_term_liab = any(
                re.search(r"long[\-\s]?term liabilities", " | ".join(v for _, v in vals), re.I)
                for vals in row_texts.values()
            )
            if not has_long_term_liab:
                continue
            for vals in row_texts.values():
                row_join = " | ".join(v for _, v in sorted(vals))
                if re.fullmatch(r"\s*Debt\s*(\|.*)?", row_join, re.I) or re.search(r"\|\s*Debt\s*\|", row_join, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
