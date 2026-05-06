def rule_page1_h1_with_company_root_path(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose text is reused as a root in descendant path_text values."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            reused = False
            for s in texts:
                path = (s.get("structure", {}).get("path_text", "") or "")
                if path.startswith(txt + " |"):
                    reused = True
                    break
            if reused:
                out.append(span)
        return out
    except Exception:
        return []
