def rule_amcor_merger_symbol_exchange(doc: dict) -> list[dict]:
    """Retrieve Amcor business-merger span stating NYSE symbol AMCR."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = f"{path} {txt}".lower()
            if span.get("label") in {"text", "section_header"} and (
                "bemis company, inc. merger" in low or ("nyse" in low and "amcr" in low and "new york stock exchange" in low)
            ):
                out.append(span)
        return out
    except Exception:
        return []

