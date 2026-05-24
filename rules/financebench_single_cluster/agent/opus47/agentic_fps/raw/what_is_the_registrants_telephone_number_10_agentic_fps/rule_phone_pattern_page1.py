import re


def rule_phone_pattern_page1(doc: dict) -> list[dict]:
    '''Page 1 spans containing a US/international phone-number pattern.'''
    pat = re.compile(
        r'(?:\+\d{1,3}\s?\d{2,4}\s?\d{6,8})'
        r'|(?:\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})'
        r'|(?:\b[1-9]\d{9}\b)'
    )
    out = []
    for s in doc.get("texts", []):
        if s.get("page_no") != 1:
            continue
        t = s.get("text", "")
        if pat.search(t):
            out.append(s)
    return out
