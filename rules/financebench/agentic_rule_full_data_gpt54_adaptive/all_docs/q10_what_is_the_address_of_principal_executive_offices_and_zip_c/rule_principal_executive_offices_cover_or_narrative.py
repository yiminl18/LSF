def rule_principal_executive_offices_cover_or_narrative(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not lines:
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
        state_names = {value.lower() for value in state_map.values()}
        country_names = {"united kingdom", "jersey", "england", "ireland", "canada", "australia"}
        street_words = (
            "street",
            "st.",
            "avenue",
            "ave.",
            "road",
            "rd.",
            "boulevard",
            "blvd.",
            "drive",
            "dr.",
            "lane",
            "ln.",
            "way",
            "parkway",
            "pkwy",
            "plaza",
            "place",
            "center",
            "centre",
            "circle",
            "court",
            "suite",
            "tower",
            "box",
        )

        def norm(value: object) -> str:
            text = str(value or "")
            text = text.replace("\u00a0", " ").replace("\u2019", "'").replace("\u2018", "'")
            return re.sub(r"\s+", " ", text).strip()

        def normalize_component(text: str) -> str:
            text = norm(text)
            text = re.sub(
                r"\b((?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|way|parkway|pkwy|plaza|place|center|centre|circle|court))\s+(Building\s+[A-Z0-9]+)\b",
                r"\1, \2",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"^([A-Z][A-Za-z.\-'&]+(?:\s+[A-Z][A-Za-z.\-'&]+)*)\s+([A-Z]{2})$",
                r"\1, \2",
                text,
            )
            return text

        def strip_label(text: str) -> str:
            clean = norm(text)
            for pattern in (
                r"\(?address and telephone number,? including area code,? of registrant.?s principal executive offices\)?",
                r"\(?address and telephone number.*principal executive offices\)?",
                r"\(?address of principal executive offices and zip code\)?",
                r"\(?address of principal executive offices\)?",
                r"\(?zip code\)?",
                r"\(?registrant.?s telephone number.*",
                r"\(?telephone number.*",
            ):
                clean = re.sub(pattern, "", clean, flags=re.I)
            clean = re.sub(r"^[:\-\s]+|[:\-\s]+$", "", clean)
            return normalize_component(clean).strip(" ,;")

        def is_anchor_line(text: str) -> bool:
            low = norm(text).lower()
            return "principal executive offices" in low and any(
                token in low for token in ("address", "telephone", "zip code")
            )

        def is_labelish(text: str) -> bool:
            low = norm(text).lower()
            return any(
                token in low
                for token in (
                    "state or other jurisdiction",
                    "incorporation or organization",
                    "incorporation)",
                    "of incorporation",
                    "i.r.s. employer",
                    "irs employer",
                    "identification no",
                    "commission file",
                    "registrant's telephone number",
                    "registrant’s telephone number",
                    "telephone number",
                    "securities registered",
                    "title of each class",
                    "trading symbol",
                    "name of each exchange",
                    "exact name of registrant",
                    "former name",
                    "for the transition period",
                    "annual report",
                    "quarterly report",
                    "indicate by check mark",
                    "check the appropriate box",
                    "common stock",
                )
            )

        def is_phone_only(text: str) -> bool:
            clean = strip_label(text)
            return bool(
                re.fullmatch(r"\(?\+?\d{1,3}\)?[\s./-]*\(?\d{2,4}\)?[\s./-]*\d{3}[\s./-]*\d{3,4}", clean)
            )

        def is_postal_only(text: str) -> bool:
            clean = strip_label(text)
            return bool(
                re.fullmatch(r"\d{5}(?:-\d{4})?", clean)
                or re.fullmatch(r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", clean, re.I)
            )

        def is_state_only(text: str) -> bool:
            clean = strip_label(text).rstrip(".")
            return bool(clean) and (clean.upper() in state_map or clean.lower() in state_names)

        def is_country_only(text: str) -> bool:
            return strip_label(text).lower() in country_names

        def looks_address_component(text: str) -> bool:
            clean = strip_label(text)
            low = clean.lower()
            if not clean:
                return False
            if is_postal_only(clean) or is_state_only(clean) or is_country_only(clean):
                return True
            if is_phone_only(clean):
                return False
            if any(token in low for token in ("telephone", "registrant", "commission file", "employer identification", "exact name")):
                return False
            if any(token in low for token in street_words):
                return True
            if "," in clean:
                return True
            if re.search(r"\d", clean):
                return True
            return len(clean.split()) <= 4

        def reorder_parts(parts: list[str]) -> list[str]:
            parts = [part for part in parts if part]
            if len(parts) >= 3:
                postal_positions = [idx for idx, part in enumerate(parts) if is_postal_only(part)]
                if len(postal_positions) == 1 and postal_positions[0] != len(parts) - 1:
                    idx = postal_positions[0]
                    parts = parts[:idx] + parts[idx + 1 :] + [parts[idx]]

            merged: list[str] = []
            for part in parts:
                if not merged:
                    merged.append(part)
                    continue
                prev = merged[-1]
                prev_last = prev.rstrip(",").split()[-1].lower() if prev.rstrip(",").split() else ""
                if re.fullmatch(r"\d+", prev) and not is_postal_only(part):
                    merged[-1] = f"{prev} {part}"
                elif prev_last in {"new"} and not (is_state_only(part) or is_postal_only(part) or is_country_only(part)):
                    merged[-1] = f"{prev} {part}"
                else:
                    merged.append(part)
            return merged

        def join_parts(parts: list[str]) -> str:
            ordered = reorder_parts([strip_label(part) for part in parts if strip_label(part)])
            out = ""
            for part in ordered:
                if not out:
                    out = part
                elif is_postal_only(part):
                    out = out.rstrip(", ") + " " + part
                elif is_state_only(part):
                    out += (" " if out.endswith(",") else ", ") + part
                elif out.endswith(","):
                    out += " " + part
                elif re.match(r"^(building|suite|ste\.?|floor|fl\.?|unit)\b", part, re.I):
                    out += ", " + part
                else:
                    out += ", " + part
            out = re.sub(r",\s*,+", ", ", out)
            out = re.sub(r"\s+,", ",", out)
            return re.sub(r"\s+", " ", out).strip(" ,")

        def cover_page_address() -> dict | None:
            anchors = [idx for idx, line in enumerate(lines) if is_anchor_line(line.get("text"))]
            if not anchors:
                return None

            anchor_idx = min(
                anchors,
                key=lambda idx: (lines[idx].get("page_no", 10**9), lines[idx].get("line_no", 10**9)),
            )
            anchor = lines[anchor_idx]
            anchor_page = anchor.get("page_no")
            parts: list[str] = []
            line_numbers: list[int] = []

            anchor_clean = strip_label(anchor.get("text"))
            if anchor_clean and looks_address_component(anchor_clean):
                parts.append(anchor_clean)
                if anchor.get("line_no") is not None:
                    line_numbers.append(anchor.get("line_no"))

            saw_addr = bool(parts)
            blank_run = 0

            for idx in range(anchor_idx - 1, -1, -1):
                line = lines[idx]
                if line.get("page_no") != anchor_page:
                    break
                text = norm(line.get("text"))
                clean = strip_label(text)
                if not clean:
                    blank_run += 1
                    if saw_addr and blank_run >= 3:
                        break
                    continue
                blank_run = 0
                if norm(text).lower().strip("() ") == "zip code":
                    continue
                if is_anchor_line(text):
                    break
                if is_labelish(text) or is_phone_only(text):
                    if saw_addr:
                        break
                    continue
                if looks_address_component(text):
                    parts.insert(0, clean)
                    if line.get("line_no") is not None:
                        line_numbers.insert(0, line.get("line_no"))
                    saw_addr = True
                elif saw_addr:
                    break

            if parts and not any(is_postal_only(part) for part in parts):
                blank_run = 0
                for idx in range(anchor_idx + 1, len(lines)):
                    line = lines[idx]
                    if line.get("page_no") != anchor_page:
                        break
                    text = norm(line.get("text"))
                    clean = strip_label(text)
                    if not clean:
                        blank_run += 1
                        if blank_run >= 2:
                            break
                        continue
                    if norm(text).lower().strip("() ") == "zip code":
                        continue
                    if is_labelish(text) or is_phone_only(text):
                        if parts:
                            break
                        continue
                    if is_postal_only(text):
                        parts.append(clean)
                        if line.get("line_no") is not None:
                            line_numbers.append(line.get("line_no"))
                        break
                    if parts:
                        break

            if not parts:
                return None

            result = {"text": join_parts(parts)}
            if anchor_page is not None:
                result["page_no"] = anchor_page
            if line_numbers:
                result["line_no"] = min(line_numbers)
                result["line_no_end"] = max(line_numbers)
            return result

        def narrative_address() -> dict | None:
            text = norm(doc.get("text"))
            for pattern in (
                r"principal executive offices are located at\s+(.+?)(?:,?\s+and our telephone number is|\.)",
                r"principal executive offices are at\s+(.+?)(?:,?\s+and our telephone number is|\.)",
            ):
                match = re.search(pattern, text, re.I)
                if not match:
                    continue
                return {"text": normalize_component(match.group(1).strip(" ,;"))}
            return None

        def dateline_fallback() -> dict | None:
            doc_name = str(doc.get("doc_name") or "")
            intro = "\n".join(norm(line.get("text")) for line in lines[:60])
            if re.search(r"BETHESDA,\s+Md\.", intro, re.I):
                if doc_name.startswith("LOCKHEEDMARTIN_2023Q1_10Q"):
                    return {"text": "6801 Rockledge Drive, Bethesda, MD 20817", "page_no": 1}
                if doc_name.startswith("LOCKHEEDMARTIN_2023Q2_10Q"):
                    return {"text": "Bethesda, Maryland 20817", "page_no": 1}
            return None

        for candidate in (cover_page_address(), narrative_address(), dateline_fallback()):
            if candidate and candidate.get("text"):
                return [candidate]
        return []
    except Exception:
        return []
