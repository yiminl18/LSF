def rule_page1_parent_path_of_state_block(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text is the parent company heading for state/EIN/address blocks."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_paths = set()
        for span in texts:
            low = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "state or other jurisdiction of incorporation" in low or "i.r.s. employer identification" in low or "irs employer identification" in low or "address of principal executive offices" in low:
                path = span.get("structure", {}).get("path_text")
                if path:
                    company_paths.add(path.split(" | ")[0].strip())
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and txt in company_paths:
                out.append(span)
        return out
    except Exception:
        return []
