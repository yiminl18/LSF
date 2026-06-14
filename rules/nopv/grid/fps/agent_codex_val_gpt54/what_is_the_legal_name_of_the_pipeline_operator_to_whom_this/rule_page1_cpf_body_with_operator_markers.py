def rule_page1_cpf_body_with_operator_markers(doc: dict) -> list[dict]:
    """Return the first page-1 post-CPF body span when it also names the operator."""
    try:
        import re

        texts = doc.get("texts", [])
        legal_with_alias_re = re.compile(
            r"\b(?:City of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}|"
            r"[A-Z][A-Za-z&().,-]+(?:\s+[A-Z][A-Za-z&().,-]+){0,8}\s+"
            r"(?:LLC|L\.L\.C\.?|LP|L\.P\.?|Inc\.?|Corp\.?|Corporation|Company|"
            r"Midstream|Utilities|Holdings|Terminals|Aviation))\s*"
            r"\([A-Za-z0-9&.\- ]+\)"
        )
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "cpf" not in ((span.get("text") or "").lower()):
                continue
            for nxt in texts[i + 1:]:
                if nxt.get("page_no") != 1:
                    break
                if nxt.get("label") != "text":
                    continue
                text = (nxt.get("text") or "").strip()
                low = text.lower()
                if not text:
                    continue
                if any(k in low for k in ["inspected", "inspection", "investigation"]) and legal_with_alias_re.search(text):
                    return [nxt]
                break
        return []
    except Exception:
        return []
