def rule_page12_cover_street_line_before_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 street-line address fragments in the window before the offices label."""
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
        zip_only_re = re.compile(r"(?i)^(?:\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})$")
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
        for idx, span in enumerate(texts):
            if (span.get("page_no") or 99) > 2:
                continue
            if not landmark_re.search(span.get("text") or ""):
                continue
            for j in range(max(0, idx - 6), idx):
                candidate = texts[j]
                raw = (candidate.get("text") or "").strip()
                if (candidate.get("page_no") or 99) > 2 or not raw:
                    continue
                if blocked_re.search(raw) or mostly_phone(raw):
                    continue
                if not any(ch.isdigit() for ch in raw) or zip_only_re.fullmatch(raw):
                    continue
                if street_re.search(raw):
                    hits.append(candidate)
        return hits
    except Exception:
        return []
