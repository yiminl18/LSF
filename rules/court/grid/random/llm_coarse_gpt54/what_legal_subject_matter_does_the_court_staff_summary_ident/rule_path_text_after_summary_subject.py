def rule_path_text_after_summary_subject(doc: dict) -> list[dict]:
    """Match spans whose path_text itself is the likely subject heading under SUMMARY."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").strip()
            txt = (span.get("text") or "").strip()
            if not path:
                continue
            up = path.upper()
            if "SUMMARY" in up:
                continue
            if any(bad in up for bad in ["COUNSEL", "OPINION", "BACKGROUND"]):
                continue
            if span.get("label") == "section_header" and txt == path:
                out.append(span)
        return out
    except Exception:
        return []
