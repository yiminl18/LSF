def rule_principal_offices_cover_window(doc: dict) -> list[dict]:
    """Return the first-page cover-page block around the principal executive offices address label."""
    try:
        import re

        records = (
            doc.get("lines")
            or doc.get("paragraphs")
            or doc.get("pages")
            or doc.get("texts")
            or []
        )
        if not records:
            return []

        def _page_num(item: dict):
            page = item.get("page_no")
            try:
                return int(page)
            except Exception:
                return page if page is not None else 10**9

        def _ord_num(item: dict):
            for key in ("line_no", "paragraph_no", "page_no"):
                val = item.get(key)
                try:
                    return int(val)
                except Exception:
                    continue
            return 10**9

        ordered = sorted(enumerate(records), key=lambda pair: (_page_num(pair[1]), _ord_num(pair[1]), pair[0]))

        numeric_pages = [p for _, item in ordered if isinstance((p := _page_num(item)), int)]
        first_page = min(numeric_pages) if numeric_pages else 1

        def _text(item: dict) -> str:
            return " ".join(
                str(item.get(key) or "")
                for key in ("text", "text_span")
                if item.get(key)
            ).strip()

        def _is_match(text: str) -> bool:
            low = text.lower()
            if "address of principal executive offices" in low:
                return True
            if "address and telephone number" in low and "principal executive offices" in low:
                return True
            if "address of principal executive offices and zip code" in low:
                return True
            if "zip code" in low and "principal executive offices" in low:
                return True
            if re.search(
                r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9&'().,\-\/ ]{3,},\s*[A-Za-z .'\-]+(?:\s+\d{5}(?:-\d{4})?)?\b",
                text,
            ):
                return True
            if re.search(
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+[A-Za-z0-9][A-Za-z0-9&'().,\-\/ ]{3,},\s*[A-Za-z .'\-]+(?:\s+\d{5}(?:-\d{4})?)?\b",
                low,
            ):
                return True
            return False

        def _is_candidate_line(text: str) -> bool:
            if not text:
                return False
            low = text.lower()
            if "address" in low or "telephone" in low or "zip code" in low:
                return False
            if re.search(
                r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9&'().,\-\/ ]{2,},\s*[A-Za-z .'\-]+(?:\s+\d{5}(?:-\d{4})?)?\b",
                text,
            ):
                return True
            if re.search(
                r"\b[A-Z][A-Za-z .'\-]+(?:,\s*|\s+)(?:[A-Z]{2}|[A-Za-z][A-Za-z .'\-]+)(?:\s+\d{5}(?:-\d{4})?)?\b",
                text,
            ):
                return True
            return False

        anchor = None
        for idx, item in ordered:
            if _page_num(item) != first_page:
                continue
            text = _text(item)
            if text and _is_match(text):
                anchor = idx
                break

        if anchor is None:
            return []

        start = max(0, anchor - 4)
        end = min(len(ordered), anchor + 3)
        keep = []
        fallback = []
        for j in range(start, end):
            if _page_num(ordered[j][1]) != first_page:
                continue
            idx, item = ordered[j]
            text = _text(item)
            fallback.append(idx)
            if _is_candidate_line(text):
                keep.append(idx)

        if not keep:
            keep = fallback

        return [records[i] for i in keep]
    except Exception:
        return []
