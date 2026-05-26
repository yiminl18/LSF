def rule_principal_offices_narrative_location(doc: dict) -> list[dict]:
    """Return narrative sentences that explicitly state where the principal executive offices are located."""
    try:
        import re

        full_text = " ".join((doc.get("text") or "").split())
        if not full_text:
            return []

        phrases = (
            "our principal executive offices are located at",
            "our principal executive offices are located in",
            "principal executive offices are located at",
            "principal executive offices are located in",
            "principal executive offices and main research facilities",
            "principal executive offices are",
        )

        low = full_text.lower()
        if "principal executive offices" not in low:
            return []
        if not any(p in low for p in phrases):
            return []
        if "located" not in low and "main research facilities" not in low:
            return []

        m = re.search(
            r"principal executive offices.*?located\s+(?:at|in(?: the)?)\s+(.+?)(?:,?\s+and\b|(?:\.\s)|$)",
            full_text,
            flags=re.IGNORECASE,
        )
        if not m:
            return []

        combined = m.group(1).strip()
        combined = re.sub(r"^\bthe\s+", "", combined, flags=re.IGNORECASE)
        combined = re.sub(r"\bmetropolitan area\.?$", "", combined, flags=re.IGNORECASE).strip(" ,")
        combined = " ".join(combined.split())
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

        out = {"text": combined}
        if doc.get("pages"):
            out["page_no"] = doc["pages"][0].get("page_no")
        return [out]
    except Exception:
        return []
