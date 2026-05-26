import re


_DUE_COURSE_RE = re.compile(
    r"\b(?:an opinion will issue in due course|the court will file a new opinion in due course)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", "")).strip()


def _due_course_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.?!])\s+", _norm(text))
    for part in parts:
        if _DUE_COURSE_RE.search(part):
            return part.strip()
    return ""


def _build_due_course_snippet(text: str) -> str:
    norm_text = _norm(text)
    parts = [part.strip() for part in re.split(r"(?<=[.?!])\s+", norm_text) if part.strip()]
    for idx, part in enumerate(parts):
        low = part.lower()
        if "oral argument will be scheduled by separate order" in low and "the court will file a new opinion in due course" in low:
            return part
        if "an opinion will issue in due course" in low:
            if idx > 0 and "heard oral argument" in parts[idx - 1].lower():
                return f"{parts[idx - 1]} {part}".strip()
            return part
        if "the court will file a new opinion in due course" in low:
            return part
    return ""


def rule_panel_disposition_due_course_order(doc: dict) -> list[dict]:
    try:
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        for para in paragraphs:
            para_text = para.get("text") or ""
            if not _DUE_COURSE_RE.search(_norm(para_text)):
                continue

            snippet = _build_due_course_snippet(para_text)
            if not snippet:
                continue

            span = {"text": f"Final disposition: {snippet}"}
            for key in ("page_no", "paragraph_no", "paragraph_index"):
                value = para.get(key)
                if value is not None:
                    span[key] = value
            return [span]

        return []
    except Exception:
        return []
