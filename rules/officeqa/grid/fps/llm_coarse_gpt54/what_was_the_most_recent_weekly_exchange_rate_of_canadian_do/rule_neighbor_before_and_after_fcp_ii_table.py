def rule_neighbor_before_and_after_fcp_ii_table(doc: dict) -> list[dict]:
    """Match FCP-II tables and their immediate neighboring spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("label") == "table" and ("fcp-ii-" in txt or "fcp-1i-" in txt):
                for j in range(max(0, i - 1), min(len(texts), i + 2)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
