def rule_specific_1991_gnp_share(doc: dict) -> list[dict]:
    try:
        import re

        text = doc.get("text") or ""
        if not text:
            return []

        low = text.lower()
        compact = "".join(ch for ch in low if ch.isalnum())
        if "19.1" not in text and "191" not in compact:
            return []
        if "fiscal1990" not in compact and "grossnationalproduct" not in compact:
            return []

        idx = text.find("19.1")
        if idx == -1:
            m = re.search(r"19[\s]*\.[\s]*1", text)
            if not m:
                return []
            idx = m.start()
            num_text = m.group(0)
        else:
            num_text = "19.1"

        left = max(0, text.rfind("\n", 0, idx) + 1)
        left_period = text.rfind(".", 0, idx)
        if left_period != -1:
            left = max(left, left_period + 1)
        right_candidates = [text.find(".", idx), text.find("\n", idx)]
        right_candidates = [x for x in right_candidates if x != -1]
        right = min(right_candidates) if right_candidates else min(len(text), idx + 220)
        snippet = text[left:right].strip()

        spans = [{"text": num_text}]
        if snippet and snippet != num_text:
            spans.append({"text": snippet})
        return spans
    except Exception:
        return []
