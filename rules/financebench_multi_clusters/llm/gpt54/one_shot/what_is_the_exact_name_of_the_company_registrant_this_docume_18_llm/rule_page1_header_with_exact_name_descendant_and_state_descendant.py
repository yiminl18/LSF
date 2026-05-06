def rule_page1_header_with_exact_name_descendant_and_state_descendant(doc: dict) -> list[dict]:
    """Match page-1 H1 headers with both exact-name and state/jurisdiction descendants nearby."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            exact = False
            state = False
            for j in range(i + 1, min(i + 20, len(texts))):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                t = nxt.get("text") or ""
                if "(Exact name of registrant as specified in its charter)" in t:
                    exact = True
                if "State or other jurisdiction of incorporation" in t or "State or other jurisdiction of incorporation or organization" in t:
                    state = True
            if exact and state:
                out.append(span)
        return out
    except Exception:
        return []
