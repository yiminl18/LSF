def rule_page12_split_locality_fragments_before_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 city/state fragments that appear shortly before the principal executive offices cover label."""
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

        state_country_re = re.compile(
            r"\b(?:[A-Z]{2}|california|washington|minnesota|oregon|illinois|maryland|virginia|new york|bristol|united kingdom)\b"
        )

        def looks_locality_fragment(text: str) -> bool:
            raw = (text or "").strip()
            norm = normalize(raw)
            if not raw or len(raw) > 90 or raw.startswith("(") or norm in {"in", "or"}:
                return False
            if any(b in norm for b in (
                "i r s employer",
                "commission file",
                "state or other jurisdiction",
                "zip code",
                "telephone number",
                "exact name of registrant",
                "trading symbol",
                "name of each exchange",
            )):
                return False
            if re.fullmatch(r"\d{4,}(?:-\d+)?", raw):
                return False
            return "," in raw or bool(state_country_re.search(raw))

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            if (span.get("page_no") or 99) > 2 or not landmark_re.search(span.get("text") or ""):
                continue
            for j in range(max(0, idx - 6), idx):
                candidate = texts[j]
                if (candidate.get("page_no") or 99) <= 2 and looks_locality_fragment(candidate.get("text") or ""):
                    hits.append(candidate)
        return hits
    except Exception:
        return []
