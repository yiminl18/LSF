def rule_report_type_with_exchange_act_reference(doc: dict) -> list[dict]:
    """Match annual/quarterly/current report spans that also reference the Securities Exchange Act of 1934."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = span.get("text") or ""
            u = t.upper()
            if "SECURITIES EXCHANGE ACT OF 1934" in u and any(k in u for k in ["ANNUAL REPORT", "QUARTERLY REPORT", "CURRENT REPORT"]):
                out.append(span)
        return out
    except Exception:
        return []
