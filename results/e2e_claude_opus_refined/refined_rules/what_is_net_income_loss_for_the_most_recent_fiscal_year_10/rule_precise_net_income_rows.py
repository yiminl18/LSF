def rule_precise_net_income_rows(doc: dict) -> list[dict]:
    """Retrieve precise net income rows/text in results-of-operations or selected-financial-data sections."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            hay = f"{path} {text}".lower()
            if (
                "consolidated results of operations" in hay
                or "selected financial data" in hay
                or "2022 results" in hay
                or "2021 results" in hay
                or "2020 results" in hay
                or "2019 results" in hay
                or "2018 results" in hay
                or "2017 results" in hay
                or "2016 results" in hay
            ) and (
                "net income" in hay
                or "net earnings" in hay
                or "net income attributable" in hay
                or "net earnings attributable" in hay
            ):
                out.append(span)
        return out
    except Exception:
        return []

