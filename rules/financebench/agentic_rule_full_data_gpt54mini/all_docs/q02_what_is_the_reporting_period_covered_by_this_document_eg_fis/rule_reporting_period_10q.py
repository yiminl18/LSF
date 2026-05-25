import re


def rule_reporting_period_10q(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines", [])[:60]

        for line in lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if re.search(r"\bquarterly\s+period\s+ended\b", text, re.IGNORECASE):
                match = re.search(r"\bquarterly\s+period\s+ended\b[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b", text, re.IGNORECASE)
                if match:
                    return [
                        {
                            "text": f"quarterly period ended {match.group(1)}",
                            "page_no": line.get("page_no"),
                            "line_no": line.get("line_no"),
                        }
                    ]

            quarter_match = re.search(r"\b(first|second|third|fourth)\s+quarter\s+(\d{4})\b", text, re.IGNORECASE)
            if quarter_match and re.search(r"\b(reports?|reported|financial results|net sales)\b", text, re.IGNORECASE):
                quarter = quarter_match.group(1).lower()
                return [
                    {
                        "text": f"{quarter} quarter {quarter_match.group(2)}",
                        "page_no": line.get("page_no"),
                        "line_no": line.get("line_no"),
                    }
                ]

        full_text = "\n".join((line.get("text") or "") for line in lines)
        match = re.search(r"\bquarterly\s+period\s+ended\b[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b", full_text, re.IGNORECASE | re.DOTALL)
        if match:
            return [{"text": f"quarterly period ended {match.group(1)}"}]

        quarter_match = re.search(r"\b(first|second|third|fourth)\s+quarter\s+(\d{4})\b", full_text, re.IGNORECASE)
        if quarter_match and re.search(r"\b(reports?|reported|financial results|net sales)\b", full_text, re.IGNORECASE):
            quarter = quarter_match.group(1).lower()
            return [{"text": f"{quarter} quarter {quarter_match.group(2)}"}]

        return []
    except Exception:
        return []
