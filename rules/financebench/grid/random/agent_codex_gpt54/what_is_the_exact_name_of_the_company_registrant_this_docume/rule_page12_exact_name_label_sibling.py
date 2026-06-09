def rule_page12_exact_name_label_sibling(doc: dict) -> list[dict]:
    """Match the short page-1/2 span immediately adjacent to the exact-name-of-registrant cover label."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").lower().replace("&", " and ")
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        blocked = {
            "united states securities and exchange commission",
            "securities and exchange commission",
            "form 10 k",
            "form 10 q",
            "form 8 k",
            "current report",
            "or",
            "washington d c 20549",
            "table of contents",
            "signatures",
        }

        hits: list[dict] = []
        texts = doc.get("texts", [])
        for idx, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if "exact name of registrant as specified in its charter" not in text:
                continue
            for neighbor_idx in (idx - 1, idx + 1):
                if 0 <= neighbor_idx < len(texts):
                    candidate = texts[neighbor_idx]
                    candidate_text = (candidate.get("text") or "").strip()
                    if (
                        (candidate.get("page_no") or 99) <= 2
                        and candidate.get("label") != "table"
                        and candidate_text
                        and len(candidate_text) <= 120
                        and normalize(candidate_text) not in blocked
                    ):
                        hits.append(candidate)
        return hits
    except Exception:
        return []
