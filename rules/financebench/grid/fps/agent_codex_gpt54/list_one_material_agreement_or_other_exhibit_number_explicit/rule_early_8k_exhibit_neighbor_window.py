def rule_early_8k_exhibit_neighbor_window(doc: dict) -> list[dict]:
    """Match short early-page 8-K spans around Item 9.01 exhibit headers and split exhibit rows."""
    try:
        import re

        texts = doc.get("texts", [])
        hits: list[dict] = []
        seen: set[int] = set()

        def add(idx: int) -> None:
            if 0 <= idx < len(texts) and idx not in seen:
                hits.append(texts[idx])
                seen.add(idx)

        def looks_like_exhibit_number(text: str) -> bool:
            return bool(re.fullmatch(r"\*{0,2}(?:\d{1,3})(?:\.\d+)*(?:[A-Z])?", text))

        for idx, span in enumerate(texts):
            page_no = span.get("page_no") or 0
            if page_no > 4:
                continue
            text = " ".join((span.get("text") or "").split())
            text_l = text.lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            is_anchor = (
                "item 9.01" in text_l
                or "financial statements and exhibits" in text_l
                or re.fullmatch(r"\(?d\)?\.?\s+exhibits\.?", text_l)
                or "exhibit no" in text_l
                or "description of exhibit" in text_l
                or ("item 9.01" in path and ("exhibit no" in text_l or "description of exhibit" in text_l))
            )
            if not is_anchor:
                continue

            for j in range(idx, min(len(texts), idx + 6)):
                candidate = texts[j]
                cand_page = candidate.get("page_no") or 0
                if cand_page and page_no and cand_page != page_no:
                    break
                cand_text = " ".join((candidate.get("text") or "").split())
                cand_l = cand_text.lower()
                if not cand_text or len(cand_text) > 240:
                    continue
                if re.search(r"\bsignatures?\b", cand_l):
                    break
                if (
                    j == idx
                    or re.fullmatch(r"\(?d\)?\.?\s+exhibits\.?", cand_l)
                    or "exhibit no" in cand_l
                    or "description of exhibit" in cand_l
                    or looks_like_exhibit_number(cand_text)
                    or re.match(r"^\*{0,2}(?:exhibit\s*)?\d+(?:\.\d+)*(?:[A-Z])?\s*[:.]?\s+\S", cand_text, re.IGNORECASE)
                ):
                    add(j)
                    if looks_like_exhibit_number(cand_text) and j + 1 < len(texts):
                        next_text = " ".join((texts[j + 1].get("text") or "").split())
                        next_page = texts[j + 1].get("page_no") or 0
                        if (
                            next_text
                            and len(next_text) <= 240
                            and "signature" not in next_text.lower()
                            and (not next_page or next_page == page_no)
                        ):
                            add(j + 1)
        return hits
    except Exception:
        return []
