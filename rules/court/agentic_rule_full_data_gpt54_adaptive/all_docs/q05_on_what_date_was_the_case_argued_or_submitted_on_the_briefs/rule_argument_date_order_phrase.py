import re


def rule_argument_date_order_phrase(doc: dict) -> list[dict]:
    try:
        month_pat = (
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
        )
        date_pat = rf"{month_pat}\s+\d{{1,2}},\s+\d{{4}}"
        phrase_patterns = (
            re.compile(
                rf"\bwe\s+heard\s+oral\s+argument(?:\s+in\s+this\s+case)?"
                rf"(?:\s+on\s+an\s+expedited\s+basis)?\s+on\s+({date_pat})\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\bwe\s+held\s+argument\s+on\s+the\s+merits\s+on\s+({date_pat})\b",
                re.IGNORECASE,
            ),
            re.compile(
                rf"\boral\s+argument\s+scheduled\s+for\s+({date_pat})\b",
                re.IGNORECASE,
            ),
        )

        def _intish(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def _clean(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\x0c", " ")).strip()

        lines = sorted(
            [item for item in (doc.get("lines") or []) if isinstance(item, dict)],
            key=lambda item: (
                _intish(item.get("page_no"), 10**9),
                _intish(item.get("line_no"), 10**9),
            ),
        )

        limit = min(len(lines), 120)
        for i in range(limit):
            window = " ".join(
                _clean(lines[j].get("text", ""))
                for j in range(i, min(i + 5, limit))
                if _clean(lines[j].get("text", ""))
            )
            if not window:
                continue
            for pattern in phrase_patterns:
                match = pattern.search(window)
                if not match:
                    continue
                span = {"text": match.group(1).strip()}
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
        for item in paragraphs[:40]:
            text = _clean(item.get("text", ""))
            if not text:
                continue
            for pattern in phrase_patterns:
                match = pattern.search(text)
                if not match:
                    continue
                span = {"text": match.group(1).strip()}
                if item.get("page_no") is not None:
                    span["page_no"] = item.get("page_no")
                if item.get("paragraph_no") is not None:
                    span["paragraph_no"] = item.get("paragraph_no")
                return [span]

        top_blob = (doc.get("text") or "").replace("\x0c", " ")[:16000]
        for pattern in phrase_patterns:
            match = pattern.search(top_blob)
            if match:
                return [{"text": match.group(1).strip()}]

        return []
    except Exception:
        return []
