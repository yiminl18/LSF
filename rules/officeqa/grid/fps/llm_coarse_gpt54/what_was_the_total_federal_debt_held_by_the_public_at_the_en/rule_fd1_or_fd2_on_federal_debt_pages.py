def rule_fd1_or_fd2_on_federal_debt_pages(doc: dict) -> list[dict]:
    """Match FD-1/FD-2 tables on Federal Debt pages."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "federal debt" in path and ("fd-1" in txt or "fd-2" in txt):
                out.append(span)
        return out
    except Exception:
        return []
