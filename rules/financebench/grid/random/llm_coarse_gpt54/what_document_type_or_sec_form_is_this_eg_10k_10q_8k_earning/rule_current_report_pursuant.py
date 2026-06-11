def rule_current_report_pursuant(doc: dict) -> list[dict]:
    """Match spans containing CURRENT REPORT and the Exchange Act reference, typical of 8-K covers."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").upper()
            if "CURRENT REPORT" in text or "PURSUANT TO SECTION 13 OR 15(D) OF THE SECURITIES EXCHANGE ACT OF 1934" in text:
                if "CURRENT REPORT" in text or "DATE OF REPORT" in text:
                    out.append(span)
        return out
    except Exception:
        return []
