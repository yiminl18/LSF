def rule_debt_note_tables(doc: dict) -> list[dict]:
    """Match debt-note tables that show total or non-current long-term debt amounts."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            debt_path = (
                "| debt" in path
                or path.endswith("debt")
                or "long-term debt" in path
                or "term debt" in path
                or ("note 4" in path and "debt" in path)
                or ("note 7" in path and "long-term debt" in path)
            )
            if debt_path and any(
                marker in text
                for marker in (
                    "less current",
                    "current portion",
                    "total debt",
                    "total long-term debt",
                    "term debt",
                    "long-term debt",
                )
            ):
                results.append(span)

        return results
    except Exception:
        return []
