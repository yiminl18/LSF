def rule_page1_first_body_after_cpf(doc: dict) -> list[dict]:
    """Return the first page-1 post-CPF body span when it names a legal entity."""
    try:
        import re

        texts = doc.get("texts", [])
        legal_phrase_re = re.compile(
            r"\b(?:City of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}|"
            r"[A-Z][A-Za-z&().,-]+(?:\s+[A-Z][A-Za-z&().,-]+){0,8}\s+"
            r"(?:LLC|L\.L\.C\.?|LP|L\.P\.?|Inc\.?|Corp\.?|Corporation|Company|"
            r"Midstream|Utilities|Holdings|Terminals|Aviation))\b"
        )
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "cpf" not in ((span.get("text") or "").lower()):
                continue
            for nxt in texts[i + 1:]:
                if nxt.get("page_no") != 1:
                    break
                text = (nxt.get("text") or "").strip()
                low = text.lower()
                if nxt.get("label") == "text" and text:
                    if legal_phrase_re.search(text) and ("inspect" in low or "investigation" in low):
                        return [nxt]
                    break
        return []
    except Exception:
        return []
