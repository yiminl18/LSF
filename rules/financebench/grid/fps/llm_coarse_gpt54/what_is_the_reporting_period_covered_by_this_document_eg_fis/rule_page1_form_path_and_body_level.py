def rule_page1_form_path_and_body_level(doc: dict) -> list[dict]:
    """Match page-1 Body-level spans under FORM paths that contain the answer phrase."""
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure") or {}
            if span.get("page_no") == 1 and struct.get("level") == "Body":
                path = (struct.get("path_text") or "").lower()
                text = (span.get("text") or "").lower()
                if "form " in path and any(k in text for k in [
                    "fiscal year ended", "quarterly period ended", "date of report"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
