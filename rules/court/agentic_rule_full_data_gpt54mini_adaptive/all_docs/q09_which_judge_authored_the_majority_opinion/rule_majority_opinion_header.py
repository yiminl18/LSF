import re


def rule_majority_opinion_header(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def with_meta(item: dict) -> dict:
            span = {"text": (item.get("text") or "").strip()}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        header_re = re.compile(
            r"\b(?:Amended\s+)?Opinion by Judge\b|\bPer Curiam Opinion\b|\bPer Curiam\b",
            re.IGNORECASE,
        )

        for key in ("lines", "paragraphs", "pages"):
            items = [item for item in (doc.get(key) or []) if isinstance(item, dict)]
            for idx, item in enumerate(items[:120]):
                text = norm(item.get("text"))
                if not text:
                    continue
                if header_re.search(text):
                    if "per curiam" in text.lower():
                        span = with_meta(item)
                        span["text"] = "The majority opinion was authored per curiam."
                        return [span]
                    return [with_meta(item)]

        full_text = doc.get("text") or ""
        m = re.search(
            r"\b(?:Amended\s+)?Opinion by Judge\b[^\n;]*",
            full_text,
            flags=re.IGNORECASE,
        )
        if m:
            return [{"text": m.group(0).strip()}]

        m = re.search(r"\bPer Curiam Opinion\b", full_text, flags=re.IGNORECASE)
        if m:
            return [{"text": m.group(0).strip()}]

        m = re.search(r"^\s*Per Curiam\s*$", full_text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return [{"text": m.group(0).strip()}]

        return []
    except Exception:
        return []
