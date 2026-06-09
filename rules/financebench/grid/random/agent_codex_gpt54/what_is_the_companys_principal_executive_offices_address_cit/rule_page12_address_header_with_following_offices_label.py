def rule_page12_address_header_with_following_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 address header spans whose following sibling is the principal executive offices label."""
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

        def looks_address_header(span: dict) -> bool:
            raw = (span.get("text") or "").strip()
            norm = normalize(raw)
            if not raw or len(raw) > 180 or raw.startswith("("):
                return False
            if any(b in norm for b in (
                "i r s employer",
                "commission file",
                "state or other jurisdiction",
                "zip code",
                "telephone number",
                "trading symbol",
                "name of each exchange",
            )):
                return False
            if re.fullmatch(r"[\d\-\(\)\s]+", raw):
                return False
            path = normalize(((span.get("structure") or {}).get("path_text") or "").split("|")[-1])
            return (
                any(ch.isdigit() for ch in raw)
                and ("," in raw or "united kingdom" in norm or path == norm or span.get("label") == "section_header")
            )

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts[:-1]):
            nxt = texts[idx + 1]
            if (span.get("page_no") or 99) > 2 or (nxt.get("page_no") or 99) > 2:
                continue
            if landmark_re.search(nxt.get("text") or "") and looks_address_header(span):
                hits.append(span)
        return hits
    except Exception:
        return []
