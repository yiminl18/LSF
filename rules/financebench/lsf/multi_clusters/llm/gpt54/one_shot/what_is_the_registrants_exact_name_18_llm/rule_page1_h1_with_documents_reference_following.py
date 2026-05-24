def rule_page1_h1_with_documents_reference_following(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with 'documents incorporated by reference' later in the same front-matter block."""
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
            for j in range(i + 1, min(i + 60, len(texts))):
                s = texts[j]
                if s.get("page_no") not in {1, 2}:
                    break
                if "documents incorporated by reference" in (s.get("text", "") or "").lower():
                    out.append(span)
                    break
        return out
    except Exception:
        return []
