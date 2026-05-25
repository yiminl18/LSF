import re


_SUMMARY_RE = re.compile(r"^\s*SUMMARY(?:\s*\*+)?\s*(.*)$", re.IGNORECASE)
_BODY_START_RE = re.compile(
    r"^(?:"
    r"The panel\b|"
    r"The district court\b|"
    r"The court\b|"
    r"Affirming\b|"
    r"Reversing\b|"
    r"Vacating\b|"
    r"Remanding\b|"
    r"Denying\b|"
    r"Granting\b|"
    r"Because\b|"
    r"Accordingly\b|"
    r"In this\b|"
    r"This summary\b|"
    r"We\b|"
    r"On appeal\b|"
    r"Holding\b|"
    r"The case\b|"
    r"This appeal\b"
    r")",
    re.IGNORECASE,
)


def rule_summary_topic(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def clean_topic(text: str) -> str:
            text = norm(text)
            text = re.sub(r"^[\[\(\{]+", "", text)
            text = re.sub(r"^[\*\s]+", "", text)
            text = re.sub(r"[\*\s]+$", "", text)
            return text.strip(" -")

        def sort_items(items: list[dict]) -> list[dict]:
            def key(item: dict) -> tuple[int, int, int]:
                page_no = item.get("page_no")
                line_no = item.get("line_no")
                paragraph_no = item.get("paragraph_no")
                page_key = page_no if isinstance(page_no, int) else 0
                line_key = line_no if isinstance(line_no, int) else 0
                para_key = paragraph_no if isinstance(paragraph_no, int) else 0
                return (page_key, line_key, para_key)

            return sorted([item for item in items if isinstance(item, dict)], key=key)

        def add_meta(span: dict, item: dict) -> dict:
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item and item.get(key) is not None:
                    span[key] = item[key]
            return span

        def extract_from_items(items: list[dict]) -> list[dict]:
            ordered = sort_items(items)
            for idx, item in enumerate(ordered):
                raw = norm(item.get("text") or "")
                if not raw:
                    continue
                match = _SUMMARY_RE.match(raw)
                if not match:
                    continue

                topic_parts: list[str] = []
                inline = clean_topic(match.group(1))
                if inline and not _BODY_START_RE.match(inline):
                    topic_parts.append(inline)

                for j in range(idx + 1, len(ordered)):
                    candidate = clean_topic(ordered[j].get("text") or "")
                    if not candidate:
                        if topic_parts:
                            break
                        continue
                    if _BODY_START_RE.match(candidate):
                        break
                    if topic_parts and len(topic_parts) >= 4:
                        break
                    topic_parts.append(candidate)

                topic = clean_topic(" ".join(topic_parts))
                if topic:
                    return [add_meta({"text": topic}, item)]
            return []

        for key in ("lines", "paragraphs", "pages"):
            found = extract_from_items(doc.get(key) or [])
            if found:
                return found

        return []
    except Exception:
        return []
