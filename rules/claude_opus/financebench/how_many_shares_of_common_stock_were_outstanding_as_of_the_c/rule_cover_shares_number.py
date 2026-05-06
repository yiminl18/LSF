def rule_cover_shares_number(doc: dict) -> list[dict]:
    """Match page 1-2 spans with large numbers following shares outstanding labels."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            page = span.get("page_no", 0)
            if page not in (1, 2):
                continue
            text = span.get("text", "").strip()
            numbers = re.findall(r'[\d,]{7,}', text)
            if not numbers:
                continue
            has_large_num = False
            for num_str in numbers:
                try:
                    num = int(num_str.replace(",", ""))
                    if num >= 1_000_000:
                        has_large_num = True
                        break
                except Exception:
                    continue
            if not has_large_num:
                continue
            for j in range(max(0, i - 10), i):
                prev = texts[j]
                if prev.get("page_no") != page:
                    continue
                prev_text = prev.get("text", "").lower()
                if "shares" in prev_text and ("outstanding" in prev_text or "common stock" in prev_text):
                    results.append(span)
                    break
        return results
    except Exception:
        return []
