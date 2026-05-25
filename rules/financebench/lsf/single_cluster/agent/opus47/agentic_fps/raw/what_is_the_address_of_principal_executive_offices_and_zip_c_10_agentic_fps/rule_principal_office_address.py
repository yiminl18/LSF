import re


def rule_principal_office_address(doc: dict) -> list[dict]:
    '''Spans on the cover page carrying the principal executive offices address and ZIP code: SEC-required marker labels ("...principal executive offices", "Zip Code") and the immediately preceding value spans that contain the address/ZIP.'''
    texts = doc.get("texts", []) or []
    addr_marker = re.compile(r"principal\s+executive\s+offices?", re.IGNORECASE)
    zip_marker = re.compile(r"\bzip\s+code\b", re.IGNORECASE)
    selected_idx = []
    seen = set()
    for i, sp in enumerate(texts):
        t = sp.get("text", "") or ""
        if addr_marker.search(t) or zip_marker.search(t):
            if i - 1 >= 0 and (i - 1) not in seen:
                seen.add(i - 1)
                selected_idx.append(i - 1)
            if i not in seen:
                seen.add(i)
                selected_idx.append(i)
    selected_idx.sort()
    return [texts[i] for i in selected_idx]
