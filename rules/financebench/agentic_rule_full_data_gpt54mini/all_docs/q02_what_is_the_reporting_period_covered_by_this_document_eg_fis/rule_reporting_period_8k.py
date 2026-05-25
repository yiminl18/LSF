import re


def rule_reporting_period_8k(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines", [])[:25]

        for line in lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if re.search(r"\bdate\s+of\s+report\b", text, re.IGNORECASE) and re.search(r"\bdate\s+of\s+earliest\s+event\s+reported\b", text, re.IGNORECASE):
                dates = re.findall(r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", text)
                if dates:
                    date_text = dates[-1]
                    return [
                        {
                            "text": date_text,
                            "page_no": line.get("page_no"),
                            "line_no": line.get("line_no"),
                        }
                    ]

        full_text = "\n".join((line.get("text") or "") for line in lines)
        if re.search(r"\bdate\s+of\s+report\b", full_text, re.IGNORECASE) and re.search(r"\bdate\s+of\s+earliest\s+event\s+reported\b", full_text, re.IGNORECASE):
            dates = re.findall(r"\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", full_text)
            if dates:
                return [{"text": dates[-1]}]

        match = re.search(r"\bdate\s+of\s+report\b.*?(\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b)", full_text, re.IGNORECASE | re.DOTALL)
        if match:
            return [{"text": match.group(1)}]

        return []
    except Exception:
        return []
