import re


_COLON_RE = re.compile(
    r"^\s*CPF\s*:\s*([1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*-\s*NOPV)\s*$",
    re.IGNORECASE,
)
_STANDARD_RE = re.compile(
    r"^\s*CPF\s+([1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*(?:-\s*|\s+)NOPV)\s*$",
    re.IGNORECASE,
)
_TEXT_RE = re.compile(
    r"(?:^|\n)\s*CPF\s*:\s*([1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*-\s*NOPV)\s*(?:$|\n)|"
    r"(?:^|\n)\s*(CPF\s+[1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*(?:-\s*|\s+)NOPV)\s*(?:$|\n)",
    re.IGNORECASE,
)
_ANYWHERE_RE = re.compile(
    r"CPF\s*:\s*([1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*-\s*NOPV)|"
    r"\b(CPF\s+[1-5]\s*-\s*\d{4}\s*-\s*\d{3}\s*(?:-\s*|\s+)NOPV)\b",
    re.IGNORECASE,
)


def rule_cpf_case_number(doc: dict) -> list[dict]:
    try:
        for source_key in ("lines", "paragraphs"):
            for item in doc.get(source_key) or []:
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                colon_match = _COLON_RE.match(text)
                if colon_match:
                    span = {"text": colon_match.group(1).strip()}
                    if "page_no" in item:
                        span["page_no"] = item["page_no"]
                    if "line_no" in item:
                        span["line_no"] = item["line_no"]
                    if "paragraph_no" in item:
                        span["paragraph_no"] = item["paragraph_no"]
                    return [span]

                standard_match = _STANDARD_RE.match(text)
                if standard_match:
                    span = {
                        "text": f"The CPF case number assigned to this enforcement action is {text}"
                    }
                    if "page_no" in item:
                        span["page_no"] = item["page_no"]
                    if "line_no" in item:
                        span["line_no"] = item["line_no"]
                    if "paragraph_no" in item:
                        span["paragraph_no"] = item["paragraph_no"]
                    return [span]

        full_text = (doc.get("text") or "").strip()
        if full_text:
            text_match = _TEXT_RE.search(full_text)
            if text_match:
                text = text_match.group(1) or text_match.group(2)
                if text:
                    if text_match.group(1):
                        return [{"text": text.strip()}]
                    return [
                        {
                            "text": f"The CPF case number assigned to this enforcement action is {text.strip()}",
                        }
                    ]

            anywhere_match = _ANYWHERE_RE.search(full_text)
            if anywhere_match:
                text = anywhere_match.group(1) or anywhere_match.group(2)
                if text:
                    if anywhere_match.group(1):
                        return [{"text": text.strip()}]
                    return [
                        {
                            "text": f"The CPF case number assigned to this enforcement action is {text.strip()}",
                        }
                    ]

        return []
    except Exception:
        return []
