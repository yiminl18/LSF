def rule_page12_cover_locality_fragments_before_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 city, state, or country fragments shortly before the offices label."""
    try:
        import re

        landmark_re = re.compile(
            r"address(?: and telephone number, including area code,)?(?: of .*?)?principal executive offices(?: and zip code)?",
            re.IGNORECASE,
        )
        blocked_re = re.compile(
            r"(commission file|employer identification|i\.?r\.?s\.? employer|telephone number|trading symbol|name of each exchange|exact name of registrant|state or other jurisdiction|securities registered)",
            re.IGNORECASE,
        )
        phone_re = re.compile(r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}")
        locality_re = re.compile(
            r"\b([A-Z]{2}|california|washington|minnesota|new york|illinois|united kingdom|bristol|san jose|seattle|issaquah|santa monica|warmley|chicago|st\. paul)\b",
            re.IGNORECASE,
        )

        def mostly_phone(text: str) -> bool:
            stripped = phone_re.sub("", text or "")
            stripped = re.sub(r"[^A-Za-z0-9]+", "", stripped)
            return not stripped

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            if (span.get("page_no") or 99) > 2:
                continue
            if not landmark_re.search(span.get("text") or ""):
                continue
            for j in range(max(0, idx - 4), idx):
                candidate = texts[j]
                raw = (candidate.get("text") or "").strip()
                low = raw.lower()
                if (candidate.get("page_no") or 99) > 2 or not raw:
                    continue
                if blocked_re.search(raw) or mostly_phone(raw):
                    continue
                if len(raw) > 80 or re.fullmatch(r"(?i)in|or", raw):
                    continue
                if low in {"delaware", "jersey"}:
                    continue
                if "," in raw or locality_re.search(raw):
                    hits.append(candidate)
        return hits
    except Exception:
        return []
