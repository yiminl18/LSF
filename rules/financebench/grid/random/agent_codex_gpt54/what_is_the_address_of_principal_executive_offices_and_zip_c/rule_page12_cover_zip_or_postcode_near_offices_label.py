def rule_page12_cover_zip_or_postcode_near_offices_label(doc: dict) -> list[dict]:
    """Match page-1/2 ZIP or postcode spans tied to the offices label or zip-code marker."""
    try:
        import re

        landmark_re = re.compile(
            r"address(?: and telephone number, including area code,)?(?: of .*?)?principal executive offices(?: and zip code)?",
            re.IGNORECASE,
        )
        zip_only_re = re.compile(r"(?i)^(?:\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})$")
        zip_any_re = re.compile(r"(?i)\b(?:\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b")

        texts = doc.get("texts", [])
        hits: list[dict] = []
        for idx, span in enumerate(texts):
            if (span.get("page_no") or 99) > 2:
                continue
            raw = (span.get("text") or "").strip()
            next_text = (texts[idx + 1].get("text") or "").strip() if idx + 1 < len(texts) else ""
            prev_window = " ".join((texts[j].get("text") or "").strip() for j in range(max(0, idx - 3), idx + 1))
            next_window = " ".join((texts[j].get("text") or "").strip() for j in range(idx, min(len(texts), idx + 4)))

            if (
                (zip_only_re.fullmatch(raw) or (zip_any_re.search(raw) and "zip code" in raw.lower()))
                and (
                    "zip code" in next_text.lower()
                    or "zip code" in raw.lower()
                    or landmark_re.search(prev_window)
                    or landmark_re.search(next_window)
                )
            ):
                hits.append(span)
                continue

            if landmark_re.search(raw):
                match = re.match(r"\s*([^()]{2,25})\s*\(", raw)
                if match and zip_any_re.search(match.group(1)):
                    hits.append(span)
        return hits
    except Exception:
        return []
