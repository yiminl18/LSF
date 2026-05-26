def rule_city_from_district_header(doc: dict) -> list[dict]:
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
            if not text or len(text) > 40:
                return False
            if re.search(r"\b(?:No\.|D\.C\.|Judge|Appeal|Circuit|Filed|Before|Court)\b", text, re.I):
                return False
            return bool(
                re.match(
                    r"^[A-Z][A-Za-z.'’-]*(?: (?:[A-Z][A-Za-z.'’-]*|d['’][A-Z][A-Za-z.'’-]*|de|del|la|las|los|of|the)){0,5}"
                    r"(?:, (?:[A-Z]{2}|[A-Z][A-Za-z .''’-]{1,}))?$",
                    text,
                )
            )

        lines = sorted(
            [item for item in (doc.get("lines") or []) if isinstance(item, dict)],
            key=lambda item: (
                _intish(item.get("page_no"), 10**9),
                _intish(item.get("line_no"), 10**9),
            ),
        )
        limit = min(len(lines), 220)
        top_window = " ".join(
            _clean(item.get("text", ""))
            for item in lines[:limit]
            if _clean(item.get("text", ""))
        )
        if not re.search(
            r"\b(?:reheard?\s+en\s+banc|rehearing\s+en\s+banc|petition\s+for\s+panel\s+rehearing|"
            r"petition\s+for\s+rehearing\s+en\s+banc|opinion\s+is\s+withdrawn|new\s+opinion\s+in\s+due\s+course|"
            r"amended\s+order|amended\s+opinion|order\s+and\s+amended\s+opinion|three-judge\s+panel\s+opinion\s+is\s+vacated|"
            r"order\s+published\b.*\bis\s+vacated)\b",
            top_window,
            re.I,
        ):
            return []
        if re.search(
            r"\b(?:stay\s+pending\s+appeal|administrative\s+stay|emergency\s+motion|temporary\s+restraining\s+order|"
            r"\bTRO\b|stay\s+of\s+permanent\s+injunction|motion\s+for\s+a\s+stay|stay\s+of\s+removal|"
            r"oral\s+argument\s+in\s+this\s+case)\b",
            top_window,
            re.I,
        ):
            return []

        for i in range(limit):
            candidate = _clean(lines[i].get("text", ""))
            if not _looks_like_city(candidate):
                continue

            prev_blob = " ".join(
                _clean(lines[j].get("text", ""))
                for j in range(max(0, i - 4), i)
                if _clean(lines[j].get("text", ""))
            )
            next_blob = " ".join(
                _clean(lines[j].get("text", ""))
                for j in range(i + 1, min(limit, i + 5))
                if _clean(lines[j].get("text", ""))
            )
            if not re.search(r"\bDistrict\b", prev_blob, re.I):
                continue
            if not re.search(r"\b(?:ORDER|OPINION|AMENDED)\b", next_blob, re.I):
                continue
            if re.search(r"\b(?:Argued|Submitted|Resubmitted)\b", prev_blob, re.I):
                continue

            span = {"text": f"Argument city: {candidate}"}
            if lines[i].get("page_no") is not None:
                span["page_no"] = lines[i].get("page_no")
            if lines[i].get("line_no") is not None:
                span["line_no"] = lines[i].get("line_no")
            return [span]
        return []
    except Exception:
        return []
