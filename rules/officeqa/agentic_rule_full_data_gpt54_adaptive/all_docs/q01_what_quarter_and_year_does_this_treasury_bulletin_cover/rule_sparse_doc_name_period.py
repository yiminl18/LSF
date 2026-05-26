import re


DOC_NAME_RE = re.compile(r"treasury_bulletin_(?P<year>\d{4})_(?P<month>\d{2})", re.IGNORECASE)


def rule_sparse_doc_name_period(doc: dict) -> list[dict]:
    try:
        text = doc.get("text") or ""
        non_ws = "".join(ch for ch in text if not ch.isspace())
        if len(non_ws) >= 100:
            return []

        doc_name = doc.get("doc_name") or ""
        match = DOC_NAME_RE.search(doc_name)
        if not match:
            return []

        year = match.group("year")
        month_num = int(match.group("month"))
        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }
        month_name = month_names.get(month_num)
        spans = []
        if month_name:
            spans.append({"text": f"{month_name} {year}"})
        return spans
    except Exception:
        return []
