def rule_exact_name_caption_parent_header(doc: dict) -> list[dict]:
    """Match the H1/section_header parent whose first child caption says 'Exact name of registrant as specified in its charter'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if "exact name of registrant as specified in its charter" in txt:
                path = span.get("structure", {}).get("path_text") or ""
                for cand in texts:
                    if (
                        cand.get("label") == "section_header"
                        and cand.get("structure", {}).get("level") == "H1"
                        and (cand.get("text") or "") == path
                    ):
                        out.append(cand)
        return out
    except Exception:
        return []
