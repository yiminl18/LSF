def rule_state_value_short_bold_page1(doc: dict) -> list[dict]:
    """Match short bold page-1 text spans immediately likely to be the state/jurisdiction value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and span.get("label") in {"text", "section_header"}
                and 2 <= len(text) <= 40
                and not re.search(r"form 10-|commission|securities|exchange|report|exact name|address|telephone|zip|trading|class|registered|documents incorporated", text, re.I)
                and not re.fullmatch(r"[\d\-\.,$() ]+", text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
