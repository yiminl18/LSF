def rule_page1_parent_of_state_ein_address_cluster(doc: dict) -> list[dict]:
    """Match page-1 headers that parent a cluster of state/EIN/address/telephone spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            score = 0
            for j in range(i + 1, min(i + 25, len(texts))):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                t = nxt.get("text") or ""
                if "State or other jurisdiction" in t:
                    score += 1
                if "Employer Identification No." in t:
                    score += 1
                if "Address of principal executive offices" in t or "Address and telephone number" in t:
                    score += 1
                if "Registrant" in t and "telephone number" in t:
                    score += 1
            if score >= 2:
                out.append(span)
        return out
    except Exception:
        return []
