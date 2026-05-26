import re


def rule_addressee_company_line(doc: dict) -> list[dict]:
    try:
        lines = [line for line in doc.get("lines", []) if line.get("page_no") == 1 and int(line.get("line_no", 0)) <= 45]
        if not lines:
            lines = doc.get("lines", [])[:45]
        if not lines:
            return []

        month_re = re.compile(
            r"(?i)^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"\s+\d{1,2}\s*,?\s+\d{4}$"
        )
        numeric_date_re = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
        person_re = re.compile(r"^(?:Mr|Ms|Mrs|Miss|Dr)\.?\s", re.I)
        title_re = re.compile(
            r"(?i)\b(?:president|vice president|chief|ceo|cfo|coo|manager|director|officer|"
            r"counsel|attorney|superintendent|supervisor|engineer|operations|regulatory|"
            r"compliance|integrity|general manager|evp|svp|vp)\b"
        )
        address_re = re.compile(
            r"(?i)(?:^p\.?\s*o\.?\s*box\b|^\d{1,6}\b|"
            r"\b(?:suite|ste\.?|road|rd\.?|street|st\.?|avenue|ave\.?|drive|dr\.?|"
            r"boulevard|blvd\.?|parkway|pkwy|lane|ln\.?|way|court|ct\.?|circle|cir\.?|"
            r"highway|hwy\.?|route|rt\.?)\b)"
        )
        city_state_zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\.?$")
        building_re = re.compile(r"(?i)\b(?:building|floor|room|tower|plaza)\b|\bno\.\s*\d+\b")

        date_idx = None
        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            if month_re.match(text) or numeric_date_re.match(text):
                date_idx = idx
                break
        if date_idx is None:
            return []

        block = []
        for line in lines[date_idx + 1 :]:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if text.startswith("CPF ") or text.startswith("Dear "):
                break
            block.append(line)
        if not block:
            return []

        address_idx = None
        for idx, line in enumerate(block):
            text = (line.get("text") or "").strip()
            if address_re.search(text) or city_state_zip_re.search(text):
                address_idx = idx
                break
        if address_idx is None:
            return []

        for idx in range(address_idx - 1, -1, -1):
            text = (block[idx].get("text") or "").strip()
            if not text:
                continue
            if person_re.match(text):
                continue
            if title_re.search(text):
                continue
            if building_re.search(text):
                continue
            return [
                {
                    "text": text,
                    "page_no": block[idx].get("page_no"),
                    "line_no": block[idx].get("line_no"),
                }
            ]
        return []
    except Exception:
        return []
