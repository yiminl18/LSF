def rule_10q_8k_company_identity_block(doc: dict) -> list[dict]:
    """Match company identity blocks in 10-Q/8-K covers that contain state, file number, or EIN nearby."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and len(text) > 20
                and (
                    re.search(r"commission file", text, re.I)
                    or re.search(r"i\.?r\.?s\.? employer identification", text, re.I)
                    or re.search(r"state or other jurisdiction of incorporation", text, re.I)
                )
            ):
                out.append(span)
        return out
    except Exception:
        return []
