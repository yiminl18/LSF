def rule_principal_offices_cover_location(doc: dict) -> list[dict]:
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
        state_names = {name.lower() for name in state_map.values()}

        def normalize(text: str) -> str:
            text = str(text or "")
            text = (
                text.replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u00a0", " ")
                .replace("\u202f", " ")
                .replace("\u2009", " ")
            )
            return re.sub(r"\s+", " ", text).strip(" ,;\t")

        def page_no(item: dict) -> int:
            try:
                return int(item.get("page_no") or 0)
            except Exception:
                return 0

        first_page = min((page_no(item) for item in lines if item.get("page_no") is not None), default=1)
        candidate_pages = {first_page, first_page + 1}

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
            "center",
            "suite",
            "ste.",
            "floor",
            "fl",
            "tower",
            "box",
        )
        block_keywords = (
            "exact name of registrant",
            "telephone",
            "registrant's telephone number",
            "registrant’s telephone number",
            "commission file",
            "employer identification",
            "state or other jurisdiction",
            "incorporation or organization",
            "securities registered",
            "trading symbol",
            "former name",
            "indicate by check mark",
            "table of contents",
            "annual report",
            "quarterly report",
            "current report",
            "common stock",
        )
        country_words = {
            "united kingdom",
            "england",
            "ireland",
            "canada",
            "australia",
            "japan",
            "germany",
            "france",
            "switzerland",
            "netherlands",
            "jersey",
        }
        us_zip_re = re.compile(r"\b\d{5}(?:-\d{4})?$")
        uk_postcode_re = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.I)

        def strip_postal(text: str) -> str:
            text = normalize(text)
            text = re.sub(r",?\s*\d{5}(?:-\d{4})?$", "", text)
            text = re.sub(r",?\s*[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", "", text, flags=re.I)
            return normalize(text)

        def normalize_region(region: str) -> str:
            region = normalize(region).rstrip(".")
            if not region:
                return ""
            mapped = state_map.get(region.upper())
            if mapped:
                return mapped
            return region

        def is_country_line(text: str) -> bool:
            clean = strip_postal(text)
            if not clean or "," in clean or any(ch.isdigit() for ch in clean):
                return False
            low = clean.lower()
            return low in country_words

        def is_addressish(text: str) -> bool:
            raw_clean = re.sub(r"\s+", " ", str(text or "")).strip()
            clean = normalize(text)
            if not clean:
                return False
            low = clean.lower()
            if any(token in low for token in block_keywords):
                return False
            if re.fullmatch(r"\(?\d{3}\)?[\s./-]*\d{3}[\s./-]*\d{4}", clean):
                return False
            if re.fullmatch(r"\d{2}-\d{7}", clean):
                return False
            if us_zip_re.fullmatch(clean) or uk_postcode_re.fullmatch(clean):
                return True
            if is_country_line(clean):
                return True
            if clean.upper().rstrip(".") in state_map or low.rstrip(".") in state_names:
                return True
            if raw_clean.endswith(",") and not any(ch.isdigit() for ch in clean):
                return True
            if "," in clean:
                return True
            parts = clean.split()
            if len(parts) >= 2 and not any(ch.isdigit() for ch in clean):
                tail = parts[-1].rstrip(".")
                if tail.upper() in state_map:
                    return True
            if any(word in low for word in street_words):
                return True
            if any(ch.isdigit() for ch in clean):
                return True
            return False

        def extract_location(line_text: str) -> str:
            clean = strip_postal(line_text)
            if not clean:
                return ""
            if "," in clean:
                parts = [normalize(part) for part in clean.split(",") if normalize(part)]
                if len(parts) < 2:
                    return ""
                city = parts[-2]
                region = normalize_region(parts[-1])
                if (
                    not city
                    or not region
                    or any(ch.isdigit() for ch in city)
                    or any(ch.isdigit() for ch in region)
                ):
                    return ""
                low_region = region.lower()
                if any(word in low_region for word in street_words):
                    return ""
                return f"{city}, {region}"

            parts = clean.split()
            if len(parts) < 2 or any(ch.isdigit() for ch in clean):
                return ""
            region = normalize_region(parts[-1])
            city = " ".join(parts[:-1]).strip()
            if not city or not region:
                return ""
            if region == parts[-1] and parts[-1].upper() not in state_map:
                return ""
            if any(word in city.lower() for word in street_words):
                return ""
            return f"{city}, {region}"

        anchor_idx = None
        for idx, item in enumerate(lines):
            if page_no(item) not in candidate_pages:
                continue
            anchor = normalize(item.get("text") or "")
            low = anchor.lower()
            if "principal executive offices" in low:
                anchor_idx = idx
                break
            next_text = ""
            if idx + 1 < len(lines) and page_no(lines[idx + 1]) == page_no(item):
                next_text = normalize(lines[idx + 1].get("text") or "")
            combined = f"{anchor} {next_text}".lower()
            if "principal executive offices" not in combined:
                continue
            if "principal" in low or "address" in low:
                anchor_idx = idx
            else:
                anchor_idx = idx + 1
            break

        if anchor_idx is None:
            return []

        anchor_page = page_no(lines[anchor_idx])
        collected: list[dict] = []
        for j in range(anchor_idx - 1, max(-1, anchor_idx - 8), -1):
            prev = lines[j]
            if page_no(prev) != anchor_page:
                continue
            raw_text = str(prev.get("text") or "")
            text = normalize(raw_text)
            low = text.lower()
            if not text:
                continue
            if "principal executive offices" in low:
                break
            if "zip code" in low:
                continue
            if any(token in low for token in block_keywords):
                if collected:
                    break
                continue
            if re.fullmatch(r"\d{2}-\d{7}", text):
                if collected:
                    break
                continue
            if re.search(r"\(?\d{3}\)?[\s./-]*\d{3}[\s./-]*\d{4}", text):
                if collected:
                    break
                continue
            collected.append(prev)
            if len(collected) >= 5:
                break

        if not collected:
            return []

        collected.reverse()
        useful = [entry for entry in collected if not us_zip_re.fullmatch(normalize(entry.get("text") or ""))]
        useful = [entry for entry in useful if not uk_postcode_re.fullmatch(normalize(entry.get("text") or ""))]
        if not useful:
            return []

        country = ""
        if is_country_line(useful[-1].get("text") or ""):
            country = strip_postal(useful[-1].get("text") or "")
            useful = useful[:-1]

        for entry in reversed(useful):
            candidate = extract_location(entry.get("text") or "")
            if candidate:
                if country and country.lower() not in candidate.lower():
                    candidate = f"{candidate}, {country}"
                return [
                    {
                        "text": candidate,
                        "page_no": entry.get("page_no"),
                        "line_no": entry.get("line_no"),
                    }
                ]

        if len(useful) >= 2:
            tail = strip_postal(useful[-1].get("text") or "").rstrip(",")
            prev = strip_postal(useful[-2].get("text") or "").rstrip(",")
            if prev and not any(ch.isdigit() for ch in prev):
                if tail.upper() in state_map or tail.lower() in state_names:
                    candidate = f"{prev}, {normalize_region(tail)}"
                    if country and country.lower() not in candidate.lower():
                        candidate = f"{candidate}, {country}"
                    return [
                        {
                            "text": candidate,
                            "page_no": useful[-2].get("page_no"),
                            "line_no": useful[-2].get("line_no"),
                        }
                    ]

        if country and useful:
            fallback = strip_postal(useful[-1].get("text") or "")
            if fallback and fallback != country and not any(ch.isdigit() for ch in fallback):
                return [
                    {
                        "text": f"{fallback}, {country}",
                        "page_no": useful[-1].get("page_no"),
                        "line_no": useful[-1].get("line_no"),
                    }
                ]

        return []
        return []
    except Exception:
        return []
