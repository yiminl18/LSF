def rule_long_term_debt_answer_cell_candidates(doc: dict) -> list[dict]:
    """Match table spans where the answer is likely in the numeric cell adjacent to a long-term debt row header."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append(c)
            for row, vals in by_row.items():
                header_hit = any(re.search(r"\blong[\-\s]?term debt\b|\blong[\-\s]?term borrowings\b", c.get("text", "") or "", re.I) for c in vals)
                numeric_hit = any(re.search(r"\d", c.get("text", "") or "") and not c.get("is_row_header") for c in vals)
                if header_hit and numeric_hit:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
