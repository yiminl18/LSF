def rule_page12_address_value_before_principal_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 address-value spans immediately before the principal executive offices cover label."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").lower().replace("’", "'")
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        landmark_re = re.compile(
            r"address(?: and telephone number, including area code,)?(?: of .*?)?principal executive offices(?: and zip code)?",
            re.IGNORECASE,
        )

        def looks_address_value(text: str) -> bool:
            raw = (text or "").strip()
            norm = normalize(raw)
            if not raw or len(raw) > 160 or raw.startswith("(") or norm in {"in", "or"}:
                return False
            blocked = (
                "address of principal executive offices",
                "telephone number",
                "i r s employer",
                "commission file",
                "state or other jurisdiction",
                "zip code",
                "exact name of registrant",
                "trading symbol",
                "name of each exchange",
            )
            if any(b in norm for b in blocked):
                return False
            if re.fullmatch(r"[A-Z]{2}", raw):
                return False
            if re.fullmatch(r"[\d\-\(\)\s]+", raw):
                return False
            has_locality = (
                "," in raw
                or "united kingdom" in norm
                or bool(re.search(r"\b(?:california|washington|minnesota|oregon|illinois|maryland|virginia|new york|bristol)\b", norm))
            )
            return any(ch.isdigit() for ch in raw) and has_locality

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx in range(1, len(texts)):
            span = texts[idx]
            if (span.get("page_no") or 99) > 2 or not landmark_re.search(span.get("text") or ""):
                continue
            candidate = texts[idx - 1]
            if (candidate.get("page_no") or 99) <= 2 and looks_address_value(candidate.get("text") or ""):
                hits.append(candidate)
        return hits
    except Exception:
        return []
