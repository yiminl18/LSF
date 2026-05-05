def rule_page1_after_shell_company_question(doc: dict) -> list[dict]:
    """Match spans after the shell-company checkbox block that mention outstanding shares."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_shell = False
        for span in texts:
            if span.get("page_no") not in (1, 2):
                continue
            t = (span.get("text") or "").lower()
            if "shell company" in t:
                seen_shell = True
            elif seen_shell and ("outstanding" in t or "issued and outstanding" in t):
                out.append(span)
        return out
    except Exception:
        return []
