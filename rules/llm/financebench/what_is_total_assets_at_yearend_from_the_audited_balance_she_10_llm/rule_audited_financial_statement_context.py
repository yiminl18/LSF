def rule_audited_financial_statement_context(doc: dict) -> list[dict]:
    """Match tables near independent auditor report / audited financial statement context."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            window = texts[max(0, i - 15):i + 3]
            joined = " ".join((w.get("text") or "").lower() for w in window)
            if (
                "independent registered public accounting firm" in joined
                or "independent auditor" in joined
                or "audited" in joined
            ) and ("assets" in (span.get("text") or "").lower()):
                out.append(span)
        return out
    except Exception:
        return []
