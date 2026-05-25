import re


_APPEAL_LABEL = "Appeal from the United States District Court"
_STOP_LABELS = ("SUMMARY",)


def rule_us_district_court_origin(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        def add_span(text: str, source_item: dict) -> None:
            cleaned = re.sub(r"\s+", " ", (text or "")).strip().rstrip(".,;:")
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in source_item:
                    span[field] = source_item[field]
            spans.append(span)

        def scan_items(items: list[dict]) -> None:
            limit = min(len(items or []), 120)
            for i in range(limit):
                item = items[i] or {}
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                upper = text.upper()
                if any(upper.startswith(stop) for stop in _STOP_LABELS):
                    break

                label_pos = text.find(_APPEAL_LABEL)
                if label_pos < 0:
                    continue

                remainder = text[label_pos + len(_APPEAL_LABEL):].strip()
                if remainder:
                    # Some OCR variants keep the district line on the same row.
                    if remainder.lower().startswith("for the "):
                        add_span(remainder, item)
                        continue

                for j in range(i + 1, min(limit, i + 6)):
                    next_item = items[j] or {}
                    next_text = (next_item.get("text") or "").strip()
                    if not next_text:
                        continue

                    next_upper = next_text.upper()
                    if any(next_upper.startswith(stop) for stop in _STOP_LABELS):
                        break

                    if next_text.lower().startswith("for the "):
                        add_span(next_text, next_item)
                        break

                    # A few pages may omit the literal "for the" prefix.
                    if (
                        "judge" not in next_text.lower()
                        and re.search(r"(?i)\b(?:district|circuit)\b", next_text)
                    ):
                        add_span(next_text, next_item)
                        break

        def scan_history_items(items: list[dict]) -> None:
            for i, item in enumerate(items or []):
                text = (item or {}).get("text") or ""
                lower = text.lower()
                if "appeal" not in lower or "united states district court for the" not in lower:
                    continue

                tail = text.split("United States District Court for the", 1)[1].strip()
                if tail:
                    add_span(tail.split(",")[0].strip(), item)
                else:
                    for j in range(i + 1, min(len(items or []), i + 4)):
                        next_item = items[j] or {}
                        next_text = (next_item.get("text") or "").strip()
                        if not next_text:
                            continue
                        if "judge" in next_text.lower():
                            continue
                        if re.search(r"(?i)\bdistrict\b", next_text):
                            add_span(next_text.split(",")[0].strip(), next_item)
                            break

                for j in range(i + 1, min(len(items or []), i + 60)):
                    next_item = items[j] or {}
                    next_text = (next_item.get("text") or "").strip()
                    if not next_text:
                        continue
                    if "judge" in next_text.lower():
                        continue
                    if "united states district court for the" not in next_text.lower():
                        continue
                    district_tail = next_text.split("United States District Court for the", 1)[1].strip()
                    if district_tail:
                        add_span(district_tail.split(",")[0].strip(), next_item)

        scan_items(doc.get("lines") or [])
        scan_items(doc.get("paragraphs") or [])
        scan_history_items(doc.get("lines") or [])
        scan_history_items(doc.get("paragraphs") or [])

        if not spans:
            full_text = doc.get("text") or ""
            for match in re.finditer(re.escape(_APPEAL_LABEL), full_text):
                start = match.end()
                tail = full_text[start:start + 180]
                tail = tail.lstrip(" \t\r\n")
                line = tail.splitlines()[0].strip() if tail else ""
                if line.lower().startswith("for the "):
                    add_span(line, {})

        return spans
    except Exception:
        return []
