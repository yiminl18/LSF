def rule_page1_cover_page_security_cluster(doc: dict) -> list[dict]:
    """Match dense page-1 cover-page clusters containing company identity plus Section 12(b) registration info."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            blob = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if s.get("page_no") == 1 and re.search(r"Exact name of registrant|Commission File|Employer Identification", blob, re.I):
                window = texts[max(0, i - 5):min(len(texts), i + 20)]
                joined = " ".join((w.get("text") or "") + " " + (w.get("text_span") or "") for w in window)
                if re.search(r"Section 12\(b\)|Trading Symbol|exchange on which registered|NASDAQ|NYSE|New York Stock Exchange", joined, re.I):
                    out.extend(window)
        return out
    except Exception:
        return []
