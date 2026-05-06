def rule_page1_h1_named_by_path_with_exact_name_child(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose path_text equals their text and that have a child caption about exact name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            parent_path = (span.get("structure", {}).get("path_text", "") or "").strip()
            if parent_path and parent_path != (span.get("text", "") or "").strip():
                continue
            for j in range(i + 1, min(i + 8, len(texts))):
                child = texts[j]
                if child.get("page_no") != 1:
                    break
                if "exact name of registrant" in (child.get("text", "") or "").lower():
                    out.append(span)
                    break
        return out
    except Exception:
        return []
