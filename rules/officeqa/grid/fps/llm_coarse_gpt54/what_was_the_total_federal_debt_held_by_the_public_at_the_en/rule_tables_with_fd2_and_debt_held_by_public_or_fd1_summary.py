def rule_tables_with_fd2_and_debt_held_by_public_or_fd1_summary(doc: dict) -> list[dict]:
    """Match either FD-2 Debt Held by the Public or FD-1 Summary of Federal Debt tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if ("fd-2" in txt and "debt held by the public" in txt) or ("fd-1" in txt and "summary of federal debt" in txt):
                out.append(span)
        return out
    except Exception:
        return []
