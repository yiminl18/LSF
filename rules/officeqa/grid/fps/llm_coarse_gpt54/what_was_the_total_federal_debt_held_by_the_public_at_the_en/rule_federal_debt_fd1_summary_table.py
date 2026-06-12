def rule_federal_debt_fd1_summary_table(doc: dict) -> list[dict]:
    """Match Federal Debt summary tables (FD-1) that often contain the year-end debt held by the public answer."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                "federal debt" in path
                and ("fd-1" in txt or "summary of federal debt" in txt)
            ):
                out.append(span)
            elif "summary of federal debt" in path and "table" in span.get("label", ""):
                out.append(span)
        return out
    except Exception:
        return []
