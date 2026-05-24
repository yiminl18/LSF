import re


def rule_cover_shares_outstanding(doc: dict) -> list[dict]:
    '''Cover-page spans (pages 1-5) mentioning "shares" and "outstanding"; for matches lacking a digit, also pull nearby same-page spans containing comma-separated numbers (handles Amazon-style split layouts).'''
    texts = doc.get("texts", [])
    num_re = re.compile(r"\d{1,3}(?:,\d{3})+")
    out = []
    seen = set()
    for i, span in enumerate(texts):
        pg = span.get("page_no", 0)
        if pg < 1 or pg > 5:
            continue
        text = span.get("text") or ""
        low = text.lower()
        if "shares" in low and "outstanding" in low:
            if i not in seen:
                seen.add(i)
                out.append(span)
            if not num_re.search(text):
                for j in range(max(0, i - 5), min(len(texts), i + 6)):
                    if j == i or j in seen:
                        continue
                    sib = texts[j]
                    if sib.get("page_no", 0) != pg:
                        continue
                    if num_re.search(sib.get("text") or ""):
                        seen.add(j)
                        out.append(sib)
    return out
