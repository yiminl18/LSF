def rule_tables_after_reports_of_independent_auditors(doc: dict) -> list[dict]:
    """Match tables appearing after auditor-report headings, where audited statements usually begin."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "reports of independent" in txt or "independent registered public accounting" in txt:
                for j in range(i + 1, min(i + 12, len(texts))):
                    if texts[j].get("label") == "table":
                        out.append(texts[j])
        return out
    except Exception:
        return []
