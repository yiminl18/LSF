def rule_page1_short_symbol_values(doc: dict) -> list[dict]:
    """Retrieve short page-1 uppercase symbol value spans from the cover-page registrant block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            try:
                if span.get("page_no") != 1 or span.get("label") != "text":
                    continue
                txt = (span.get("text") or "").strip()
                path = ((span.get("structure") or {}).get("path_text") or "")
                if not path:
                    continue
                if re.fullmatch(r"[A-Z]{2,5}", txt) is None:
                    continue
                if txt in {"FORM", "ITEM", "PART", "YES", "NO", "OR"}:
                    continue
                if any(k in path for k in ["INC.", "PLC", "Corporation", "COMPANY", "Company"]):
                    out.append(span)
            except Exception:
                continue
        return out
    except Exception:
        return []

