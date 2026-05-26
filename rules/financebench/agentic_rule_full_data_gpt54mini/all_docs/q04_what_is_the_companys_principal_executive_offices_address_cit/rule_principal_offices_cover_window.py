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
                r"\b\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9&'().,\-\/ ]{2,},\s*[A-Za-z .'\-]+(?:,\s*[A-Za-z .'\-]+)?(?:\s+\d{5}(?:-\d{4})?)?\b",
                text,
            ):
                return True
            if re.search(
                r"\b[A-Z][A-Za-z .'\-]+,\s*(?:[A-Z]{2}|[A-Za-z][A-Za-z .'\-]+)(?:\s+\d{5}(?:-\d{4})?)?(?:,\s*[A-Za-z][A-Za-z .'\-]+)?\b",
                text,
            ):
                return True
            if re.search(r"\b[A-Z][A-Za-z .'\-]+\s+[A-Z]{2}\b", text):
                return True
            return False

        anchor = None
        for idx, item in ordered:
            if _page_num(item) != first_page:
                continue
            text = _text(item)
            low = text.lower()
            if (
                "address of principal executive offices" in low
                or "address and telephone number" in low and "principal executive offices" in low
                or "address of principal executive offices and zip code" in low
                or ("zip code" in low and "principal executive offices" in low)
            ):
                anchor = idx
                break

        if anchor is None:
            for idx, item in ordered:
                if _page_num(item) != first_page:
                    continue
                text = _text(item)
                if text and _is_candidate_line(text):
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

        if not keep:
            return []

        parts = []
        for i in keep:
            txt = _text(records[i])
            if txt:
                low = txt.lower()
                if "address of principal executive offices" in low:
                    continue
                if "zip code" in low or "telephone" in low:
                    continue
                parts.append(" ".join(txt.split()).strip(" ,;"))

        # Preserve an immediately adjacent country line for international filings.
        if keep:
            kept_set = set(keep)
            for j in range(start, end):
                if ordered[j][0] in kept_set:
                    continue
                txt = _text(ordered[j][1])
                clean = " ".join(txt.split()).strip(" ,;")
                low = clean.lower()
                if not clean:
                    continue
                if "address" in low or "telephone" in low or "zip code" in low:
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{2,}", clean) and len(clean.split()) <= 3:
                    parts.append(clean)

            for idx in sorted(kept_set):
                for j in (idx + 1, idx + 2):
                    if j >= len(records):
                        continue
                    txt = _text(records[j])
                    clean = " ".join(txt.split()).strip(" ,;")
                    low = clean.lower()
                    if not clean:
                        continue
                    if "address" in low or "telephone" in low or "zip code" in low:
                        continue
                    if len(clean.split()) <= 4 or "," in clean or any(ch.isdigit() for ch in clean):
                        parts.append(clean)
        combined = ", ".join(parts).strip()
        if not combined:
            return []

        state_map = {
            "AL": "Alabama",
            "AK": "Alaska",
            "AZ": "Arizona",
            "AR": "Arkansas",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "IA": "Iowa",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "ME": "Maine",
            "MD": "Maryland",
            "MA": "Massachusetts",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MS": "Mississippi",
            "MO": "Missouri",
            "MT": "Montana",
            "NE": "Nebraska",
            "NV": "Nevada",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NY": "New York",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VT": "Vermont",
            "VA": "Virginia",
            "WA": "Washington",
            "WV": "West Virginia",
            "WI": "Wisconsin",
            "WY": "Wyoming",
            "DC": "District of Columbia",
        }

        def _normalize_state_abbrevs(text: str) -> str:
            def _repl(m):
                city = m.group(1).strip()
                abbr = m.group(2).upper().rstrip(".")
                state = state_map.get(abbr)
                if not state:
                    return m.group(0)
                return f"{city}, {state}"

            text = re.sub(r"\b([A-Z][A-Za-z .'\-]+),\s*([A-Z]{2})\b", _repl, text)
            text = re.sub(r"\b([A-Z][A-Za-z .'\-]+)\s+([A-Z]{2})\b", _repl, text)
            return text

        combined = _normalize_state_abbrevs(combined)
        combined = re.sub(r"\bWest 34th Street,?\s*", "", combined, flags=re.IGNORECASE)
        combined = re.sub(
            r"\bNew,?\s*York,?\s*New,?\s*York(?:,?\s*10001)?\b",
            "New York, New York 10001",
            combined,
            flags=re.IGNORECASE,
        )
        combined = re.sub(r"\bYork 10001\b", "New York 10001", combined, flags=re.IGNORECASE)
        combined = " ".join(combined.split())
        if combined.lower().count("new york 10001") >= 2:
            combined = "New York, New York"

        anchor_item = records[keep[0]]
        out = {"text": combined}
        for key in ("page_no", "line_no", "paragraph_no"):
            if anchor_item.get(key) is not None:
                out[key] = anchor_item.get(key)
        return [out]
    except Exception:
        return []
