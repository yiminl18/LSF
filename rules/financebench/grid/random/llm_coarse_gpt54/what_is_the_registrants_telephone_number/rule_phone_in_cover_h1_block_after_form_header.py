def rule_phone_in_cover_h1_block_after_form_header(doc: dict) -> list[dict]:
    """Match phone-number spans after the form header and within the company cover block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        start = 0
        for i, span in enumerate(texts):
            if re.search(r"FORM\s+8-K|FORM\s+10-K|FORM\s+10-Q", span.get("text", "") or "", re.I):
                start = i
                break
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in texts[start:start + 40]:
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if phone_re.search(span.get("text", "") or "") and not re.search(r"PART I|TABLE OF CONTENTS|INDEX", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
