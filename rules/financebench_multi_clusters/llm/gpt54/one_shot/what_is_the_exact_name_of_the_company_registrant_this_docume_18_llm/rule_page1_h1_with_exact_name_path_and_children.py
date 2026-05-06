def rule_page1_h1_with_exact_name_path_and_children(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose child path contains the H1 text and exact-name marker appears among children."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            txt = (span.get("text") or "").strip()
            if not txt:
                continue
            child_match = False
            for j in range(i + 1, min(len(texts), i + 20)):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                p = nxt.get("structure", {}).get("path_text") or ""
                if txt in p and "(Exact name of registrant as specified in its charter)" in (nxt.get("text") or ""):
                    child_match = True
                    break
            if child_match:
                out.append(span)
        return out
    except Exception:
        return []
