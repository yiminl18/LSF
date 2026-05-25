import re


def rule_opening_location_sentence(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []
        limit = min(len(lines), 80)
        markers = ("inspected", "located", "facilities", "inspection", "investigated")
        hits = []
        for i in range(limit):
            txt = (lines[i].get("text") or "").strip()
            if not txt:
                continue
            low = txt.lower()
            loc = ("," in txt and re.search(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", txt)) or re.search(r"\bin [A-Z][a-z]+(?: [A-Z][a-z]+)?\b", txt)
            prev = (lines[i - 1].get("text") or "").lower() if i else ""
            prev2 = (lines[i - 2].get("text") or "").lower() if i > 1 else ""
            if ((any(m in low for m in markers) and loc) or ((i and any(m in prev for m in markers)) or (i > 1 and any(m in prev2 for m in markers))) and loc):
                hits.append(i)
        if not hits:
            return []
        i = hits[0]
        a = max(0, i - 1)
        b = min(limit, i + 5)
        text = " ".join((lines[j].get("text") or "").strip() for j in range(a, b) if (lines[j].get("text") or "").strip())
        return [{"text": text, "page_no": lines[a].get("page_no"), "line_no": lines[a].get("line_no")}]
    except Exception:
        return []
