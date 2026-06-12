def rule_answer_page_from_contents_statutory_limit(doc: dict) -> list[dict]:
    """Use contents references to statutory limit tables to return spans from the referenced page and adjacent page."""
    import re
    try:
        pages = set()
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "contents" not in path.lower():
                continue
            if re.search(r"debt subject to statutory|status and application of statutory limitation|statutory limit", text, re.I):
                for n in re.findall(r"\b\d+\b", text):
                    try:
                        pages.add(int(n))
                    except Exception:
                        pass
        if not pages:
            return []
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if isinstance(p, int) and any(abs(p - tp) <= 1 for tp in pages):
                out.append(span)
        return out
    except Exception:
        return []
