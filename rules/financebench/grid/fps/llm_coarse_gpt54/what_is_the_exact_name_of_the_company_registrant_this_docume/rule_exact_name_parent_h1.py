def rule_exact_name_parent_h1(doc: dict) -> list[dict]:
    """Match H1/header-like spans on page 1 whose immediate child text says exact name of registrant."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            if span.get("structure", {}).get("level") not in {"H1", "Body"}:
                continue
            sid = i
            for child in texts:
                st = child.get("text", "") or ""
                if "exact name of registrant as specified in its charter" in st.lower():
                    if child.get("structure", {}).get("parent_id") == sid:
                        out.append(span)
                        break
        return out
    except Exception:
        return []
