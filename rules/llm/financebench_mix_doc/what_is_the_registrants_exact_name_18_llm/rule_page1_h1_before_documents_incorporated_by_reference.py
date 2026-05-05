def rule_page1_h1_before_documents_incorporated_by_reference(doc: dict) -> list[dict]:
    """Match page-1 H1 company headers whose section extends down to documents incorporated by reference."""
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
            found = False
            for j in range(i + 1, min(i + 80, len(texts))):
                s = texts[j]
                if s.get("page_no") not in {1, 2, 3}:
                    break
                t = (s.get("text", "") or "").lower()
                if "documents incorporated by reference" in t:
                    found = True
                    break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
