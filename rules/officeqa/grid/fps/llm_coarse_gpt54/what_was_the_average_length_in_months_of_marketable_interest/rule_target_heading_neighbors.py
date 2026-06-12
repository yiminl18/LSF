def rule_target_heading_neighbors(doc: dict) -> list[dict]:
    """Return spans immediately following a heading/title that names the target table."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if (
                "maturity distribution" in txt
                and "average length" in txt
                and span.get("label") in {"section_header", "text", "table"}
            ):
                for j in range(i + 1, min(i + 4, len(texts))):
                    out.append(texts[j])
    except Exception:
        return []
    return out
