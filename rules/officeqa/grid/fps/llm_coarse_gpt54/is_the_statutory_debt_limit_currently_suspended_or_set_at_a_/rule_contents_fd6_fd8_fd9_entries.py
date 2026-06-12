def rule_contents_fd6_fd8_fd9_entries(doc: dict) -> list[dict]:
    """Match contents-page tables listing FD/FO entries for debt subject to statutory limitation or status/application of statutory limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "contents" not in path.lower():
                continue
            if span.get("label") not in {"table", "text"}:
                continue
            if re.search(r"\b(FD|FO)[-\s]?(6|8|9)\b", text, re.I) and re.search(r"statutory (limit|limitation)|debt subject to statutory|status and application of statutory", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
