def rule_city_from_oral_argument_order_phrase(doc: dict) -> list[dict]:
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
            if not text or len(text) > 50:
                return False
            if re.search(r"\b(?:case|basis|motion|order|appeal|argument|submitted)\b", text, re.I):
                return False
            return bool(
                re.match(
                    r"^[A-Z][A-Za-z.'’-]*(?: (?:[A-Z][A-Za-z.'’-]*|d['’][A-Z][A-Za-z.'’-]*|de|del|la|las|los|of|the)){0,5}"
                    r"(?:, (?:[A-Z]{2}|[A-Z][A-Za-z .''’-]{1,}))?$",
                    text,
                )
            )

        city_pat = (
            r"[A-Z][A-Za-z.'’-]*(?: (?:[A-Z][A-Za-z.'’-]*|d['’][A-Z][A-Za-z.'’-]*|de|del|la|las|los|of|the)){0,5}"
            r"(?:, (?:[A-Z]{2}|[A-Z][A-Za-z .''’-]{1,}))?"
        )
        phrase_patterns = [
            re.compile(
                rf"\b(?:reschedule|reset|calendar(?:ed)?)\s+oral\s+argument\b.*?\bin\s+({city_pat})\b",
                re.I,
            ),
            re.compile(
                rf"\bcalendared\s+this\s+case\s+for\s+argument(?:\s+before\s+a\s+merits\s+panel)?\s+in\s+({city_pat})\b",
                re.I,
            ),
            re.compile(
                rf"\bargument(?:\s+for\s+this\s+case)?\s+is\s+set\s+for\b.*?\bin\s+({city_pat})\b",
                re.I,
            ),
            re.compile(
                rf"\bweek\s+of\s+[A-Z][a-z]+\s+\d{{1,2}},\s+\d{{4}}\s+in\s+({city_pat})\b",
                re.I,
            ),
            re.compile(
                rf"\bheard\s+oral\s+argument(?:\s+in\s+this\s+case)?\s+in\s+({city_pat})\b",
                re.I,
            ),
        ]

        lines = sorted(
            [item for item in (doc.get("lines") or []) if isinstance(item, dict)],
            key=lambda item: (
                _intish(item.get("page_no"), 10**9),
                _intish(item.get("line_no"), 10**9),
            ),
        )
        limit = min(len(lines), 260)
        for i in range(limit):
            window = " ".join(
                _clean(lines[j].get("text", ""))
                for j in range(i, min(i + 6, limit))
                if _clean(lines[j].get("text", ""))
            )
            if not window:
                continue
            for pattern in phrase_patterns:
                match = pattern.search(window)
                if not match:
                    continue
                city_text = match.group(1).strip().rstrip(".,;:")
                if not _looks_like_city(city_text):
                    continue
                span = {"text": f"Argument city: {city_text}"}
                if lines[i].get("page_no") is not None:
                    span["page_no"] = lines[i].get("page_no")
                if lines[i].get("line_no") is not None:
                    span["line_no"] = lines[i].get("line_no")
                return [span]

        paragraphs = sorted(
            [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)],
            key=lambda item: (
                _intish(item.get("page_no"), 10**9),
                _intish(item.get("paragraph_no"), 10**9),
            ),
        )
        for item in paragraphs[:80]:
            text = _clean(item.get("text", ""))
            if not text:
                continue
            for pattern in phrase_patterns:
                match = pattern.search(text)
                if not match:
                    continue
                city_text = match.group(1).strip().rstrip(".,;:")
                if not _looks_like_city(city_text):
                    continue
                span = {"text": f"Argument city: {city_text}"}
                if item.get("page_no") is not None:
                    span["page_no"] = item.get("page_no")
                if item.get("paragraph_no") is not None:
                    span["paragraph_no"] = item.get("paragraph_no")
                return [span]

        return []
    except Exception:
        return []
