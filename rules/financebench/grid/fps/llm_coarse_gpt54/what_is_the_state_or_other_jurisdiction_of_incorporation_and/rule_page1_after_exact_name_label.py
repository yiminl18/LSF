def rule_page1_after_exact_name_label(doc: dict) -> list[dict]:
    """Match the next few spans after the exact-name label, where state and EIN values often appear."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", txt, re.I):
                for cand in texts[i:i+8]:
                    ctext = (cand.get("text") or "") + " " + (cand.get("text_span") or "")
                    if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", ctext, re.I):
                        out.append(cand)
                    elif re.search(r"employer\s+identification", ctext, re.I):
                        out.append(cand)
                    elif re.fullmatch(r"\d{2}-\d{7}", (cand.get("text") or "").strip()):
                        out.append(cand)
                    elif re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", (cand.get("text") or "").strip(), re.I):
                        out.append(cand)
                break
        return out
    except Exception:
        return []
