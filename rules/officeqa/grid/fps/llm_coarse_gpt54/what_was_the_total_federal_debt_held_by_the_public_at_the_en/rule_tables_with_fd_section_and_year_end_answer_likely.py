def rule_tables_with_fd_section_and_year_end_answer_likely(doc: dict) -> list[dict]:
    """Broad high-recall rule for Federal Debt section tables likely to contain year-end totals."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "federal debt" in path and (
                "summary" in txt or "public debt" in txt or "held by the public" in txt or "debt held by the public" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
