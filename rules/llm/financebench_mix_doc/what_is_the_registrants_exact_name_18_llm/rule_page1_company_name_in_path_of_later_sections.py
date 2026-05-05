def rule_page1_company_name_in_path_of_later_sections(doc: dict) -> list[dict]:
    """Match page-1 spans whose text becomes the root path_text prefix of later sections."""
    try:
        texts = doc.get("texts", [])
        out = []
        page1 = [s for s in texts if s.get("page_no") == 1 and s.get("label") in {"text", "section_header"}]
        later_paths = [((s.get("structure", {}) or {}).get("path_text", "") or "") for s in texts if s.get("page_no", 0) >= 2]
        for span in page1:
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if not txt:
                continue
            if any(p.startswith(txt + " |") or p == txt for p in later_paths):
                if (
                    "form 10-" not in low
                    and "form 8-k" not in low
                    and "current report" not in low
                    and "securities and exchange commission" not in low
                ):
                    out.append(span)
        return out
    except Exception:
        return []
