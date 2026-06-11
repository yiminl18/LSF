def rule_item1_business_amcor_registered_and_traded(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans describing shares/notes registered and traded on an exchange under a symbol."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"item\s*1|business", path, re.I) and (
                re.search(r"traded on .* under the symbol", txt, re.I)
                or re.search(r"registered .* traded on .* under the symbol", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
