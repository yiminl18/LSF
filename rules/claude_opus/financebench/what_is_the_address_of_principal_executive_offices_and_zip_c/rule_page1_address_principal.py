def rule_page1_address_principal(doc: dict) -> list[dict]:
    """Match page 1 spans containing or near 'address of principal executive offices'."""
    try:
        results = []
        texts = doc.get("texts", [])
        seen_ids = set()

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue

            text = span.get("text", "").lower()
            text_span = span.get("text_span", "").lower()

            # Pattern 1: text_span contains the address label - span.text has the address
            if "address of principal" in text_span or "address and telephone" in text_span:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))

            # Pattern 2: text contains the address label - previous span has address
            if "address of principal" in text or "address and telephone" in text:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                for offset in [1, 2, 3]:
                    prev_idx = i - offset
                    if prev_idx >= 0 and texts[prev_idx].get("page_no") == 1:
                        prev = texts[prev_idx]
                        if id(prev) not in seen_ids:
                            prev_text = prev.get("text", "").lower()
                            if "jurisdiction" in prev_text or "i.r.s." in prev_text or "employer identification" in prev_text:
                                continue
                            results.append(prev)
                            seen_ids.add(id(prev))

            # Pattern 3: Also capture "(zip code)" patterns
            if "(zip code)" in text:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                if i > 0 and texts[i-1].get("page_no") == 1:
                    prev = texts[i-1]
                    if id(prev) not in seen_ids:
                        results.append(prev)
                        seen_ids.add(id(prev))

        return results
    except Exception:
        return []
