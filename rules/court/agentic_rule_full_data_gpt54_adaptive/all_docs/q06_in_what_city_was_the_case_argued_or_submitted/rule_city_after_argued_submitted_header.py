def rule_city_after_argued_submitted_header(doc: dict) -> list[dict]:
    try:
        import re

        def _intish(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def _clean(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\x0c", " ")).strip()

        def _looks_like_city(text: str) -> bool:
            if not text:
                return False
            if len(text) > 50:
                return False
            if re.search(
                r"\b(?:Before|Filed|Appeal|Opinion|ORDER|SUMMARY|COUNSEL|Judge|Judges|Circuit|D\.C\. No\.|No\.)\b",
                text,
                re.I,
            ):
                return False
            return bool(
                re.match(
                    r"^[A-Z][A-Za-z.'’-]*(?: (?:[A-Z][A-Za-z.'’-]*|d['’][A-Z][A-Za-z.'’-]*|de|del|la|las|los|of|the)){0,5}"
                    r"(?:, (?:[A-Z]{2}|[A-Z][A-Za-z .''’-]{1,}))?$",
                    text,
                )
            )

        def _looks_like_header(text: str) -> bool:
            if not text:
                return False
            if not re.search(r"\b(?:Argued|Submitted|Resubmitted)\b", text, re.I):
                return False
            if re.search(
                r"\b(?:Submission Vacated|Submission Withdrawn|Submission Deferred)\b",
                text,
                re.I,
            ):
                return False
            return bool(
                re.search(
                    r"\b(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
                    text,
                    re.I,
                )
                or re.search(r"\b\d{4}\b", text)
            )

        lines = sorted(
            [item for item in (doc.get("lines") or []) if isinstance(item, dict)],
            key=lambda item: (
                _intish(item.get("page_no"), 10**9),
                _intish(item.get("line_no"), 10**9),
            ),
        )
        if not lines:
            return []

        limit = min(len(lines), 420)
        for i in range(limit):
            for end in range(i + 1, min(i + 4, limit + 1)):
                combined = " ".join(
                    _clean(lines[j].get("text", ""))
                    for j in range(i, end)
                    if _clean(lines[j].get("text", ""))
                )
                if not _looks_like_header(combined):
                    continue

                inline_match = re.search(
                    r"\b(?:Argued(?:\s+and\s+Submitted)?|Submitted(?:\s+on\s+the\s+briefs)?|Resubmitted)"
                    r".*?\b\d{4}\b(?:\*+)?\s+"
                    r"([A-Z][A-Za-z.'’-]*(?: (?:[A-Z][A-Za-z.'’-]*|d['’][A-Z][A-Za-z.'’-]*|de|del|la|las|los|of|the)){0,5}"
                    r"(?:, (?:[A-Z]{2}|[A-Z][A-Za-z .''’-]{1,}))?)$",
                    combined,
                    re.I,
                )
                if inline_match:
                    city_text = inline_match.group(1).strip()
                    if _looks_like_city(city_text):
                        span = {"text": f"Argument city: {city_text}"}
                        if lines[i].get("page_no") is not None:
                            span["page_no"] = lines[i].get("page_no")
                        if lines[i].get("line_no") is not None:
                            span["line_no"] = lines[i].get("line_no")
                        return [span]

                for j in range(end, min(end + 8, limit)):
                    candidate = _clean(lines[j].get("text", ""))
                    if not candidate:
                        continue
                    if re.search(
                        r"\b(?:Argued|Submitted|Resubmitted|Submission Vacated|Submission Withdrawn|Submission Deferred)\b",
                        candidate,
                        re.I,
                    ):
                        continue
                    if re.search(r"\b(?:Filed|Before|Opinion by|SUMMARY|COUNSEL)\b", candidate, re.I):
                        break
                    if _looks_like_city(candidate):
                        span = {"text": f"Argument city: {candidate}"}
                        if lines[j].get("page_no") is not None:
                            span["page_no"] = lines[j].get("page_no")
                        if lines[j].get("line_no") is not None:
                            span["line_no"] = lines[j].get("line_no")
                        return [span]
        return []
    except Exception:
        return []
