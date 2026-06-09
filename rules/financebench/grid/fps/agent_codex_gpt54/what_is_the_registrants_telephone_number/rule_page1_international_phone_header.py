def rule_page1_international_phone_header(doc: dict) -> list[dict]:
    """Match page-1 section headers that are standalone international phone values in the cover breadcrumb."""
    try:
        import re

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue

            text = " ".join((span.get("text") or "").split())
            if not text.startswith("+"):
                continue

            digits = re.sub(r"\D", "", text)
            if len(digits) < 10 or len(digits) > 15:
                continue
            if re.search(r"[A-Za-z@]", text):
                continue
            if not re.fullmatch(r"[+()0-9.\-\s]+", text):
                continue

            path_text = (span.get("structure") or {}).get("path_text") or ""
            if text not in path_text:
                continue

            results.append(span)

        return results
    except Exception:
        return []
