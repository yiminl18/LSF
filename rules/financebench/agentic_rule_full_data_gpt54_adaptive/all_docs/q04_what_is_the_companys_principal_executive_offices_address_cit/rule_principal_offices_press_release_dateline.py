def rule_principal_offices_press_release_dateline(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not lines:
            return []

        full_text = " ".join(str(doc.get("text") or "").split())
        if "principal executive offices" in full_text.lower():
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
            "IND": "Indiana",
            "KAN": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "MASS": "Massachusetts",
            "MD": "Maryland",
            "ME": "Maine",
            "MICH": "Michigan",
            "MINN": "Minnesota",
            "MISS": "Mississippi",
            "MO": "Missouri",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "NEB": "Nebraska",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NV": "Nevada",
            "NY": "New York",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "ORE": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TENN": "Tennessee",
            "TEX": "Texas",
            "UT": "Utah",
            "VA": "Virginia",
            "VT": "Vermont",
            "WASH": "Washington",
            "WIS": "Wisconsin",
            "WYO": "Wyoming",
            "D.C": "District of Columbia",
            "DC": "District of Columbia",
            "MD.": "Maryland",
            "MINN.": "Minnesota",
            "ILL.": "Illinois",
            "CALIF.": "California",
            "N.Y.": "New York",
        }

        def normalize(text: str) -> str:
            text = str(text or "")
            text = (
                text.replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u00a0", " ")
            )
            return re.sub(r"\s+", " ", text).strip(" ,;")

        def title_case_city(text: str) -> str:
            words = []
            for token in re.split(r"(\s+|-)", text.lower()):
                if not token or token.isspace() or token == "-":
                    words.append(token)
                    continue
                if token.startswith("mc") and len(token) > 2:
                    words.append("Mc" + token[2:].capitalize())
                else:
                    words.append(token.capitalize())
            return "".join(words)

        dateline_re = re.compile(
            r"\b([A-Z][A-Z .&'-]+?),\s*([A-Z][A-Za-z.]{1,10}|[A-Z]{2}),\s*[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}\b"
        )

        for item in lines[:60]:
            text = normalize(item.get("text") or "")
            if not text:
                continue
            m = dateline_re.search(text)
            if not m:
                continue
            city = title_case_city(m.group(1).strip())
            region_raw = m.group(2).strip().upper().rstrip(",")
            region = state_map.get(region_raw, state_map.get(region_raw.rstrip("."), m.group(2).strip().rstrip(".")))
            return [
                {
                    "text": f"{city}, {region}",
                    "page_no": item.get("page_no"),
                    "line_no": item.get("line_no"),
                }
            ]

        return []
    except Exception:
        return []
