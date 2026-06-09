def rule_page12_cover_address_value_before_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 address-value spans immediately before the principal executive offices label."""
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
        street_re = re.compile(
            r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|boulevard|blvd\.?|lane|ln\.?|plaza|parkway|pkwy\.?|center|centre|way|place|pl\.?|court|ct\.?|circle|cir\.?|highway|hwy\.?|building|bldg\.?|tower|terrace|ter\.?|square|sq\.?|north|south|east|west|olympic|ocean)\b",
            re.IGNORECASE,
        )

        def mostly_phone(text: str) -> bool:
            stripped = phone_re.sub("", text or "")
            stripped = re.sub(r"[^A-Za-z0-9]+", "", stripped)
            return not stripped

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx in range(1, len(texts)):
            span = texts[idx]
            if (span.get("page_no") or 99) > 2:
                continue
            if not landmark_re.search(span.get("text") or ""):
                continue
            candidate = texts[idx - 1]
            raw = (candidate.get("text") or "").strip()
            if (candidate.get("page_no") or 99) > 2 or not raw:
                continue
            if blocked_re.search(raw) or re.fullmatch(r"(?i)in|or", raw) or mostly_phone(raw):
                continue
            if any(ch.isdigit() for ch in raw) and (
                street_re.search(raw) or "," in raw or "united kingdom" in raw.lower()
            ):
                hits.append(candidate)
        return hits
    except Exception:
        return []
