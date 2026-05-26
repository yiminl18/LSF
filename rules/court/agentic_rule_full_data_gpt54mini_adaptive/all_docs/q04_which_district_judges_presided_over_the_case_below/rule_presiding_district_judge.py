import re


_ROLE_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:(?:United States\s+|U\.S\.\s+)?(?:Chief|Senior)\s+)?"
    r"(?:United States\s+|U\.S\.\s+)?District(?:\s+Judge|\s+Court)(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)
_MAGISTRATE_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:United States\s+|U\.S\.\s+)?Magistrate Judge(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)
_BANKRUPTCY_RE = re.compile(
    r"(?is)^\s*(?:Hon\.?\s*)?(?P<name>.+?)\s*,\s*"
    r"(?:United States\s+|U\.S\.\s+)?Bankruptcy Judges?(?:,\s*|\s+)Presiding\s*[\.\*\u2020]*\s*$"
)

_ROLE_HINT_RE = re.compile(
    r"(?i)\b(?:(?:chief|senior)\s+)?district judge\b|\bmagistrate judge\b|\bbankruptcy judges?\b"
)
_NAMEISH_RE = re.compile(
    r"(?i)^\s*(?:Hon\.?\s*)?[A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,6}\s*,?\s*$"
)
_NAME_FRAGMENT_RE = re.compile(
    r"(?i)^\s*(?:Hon\.?\s*)?[A-Z][A-Za-z0-9.'’\- ]{0,120}(?:,\s*and| and|,|&)\s*$"
)
_NON_NAME_HINT_RE = re.compile(
    r"(?i)\b(?:appeal|appeals|district court|bankruptcy appellate panel|bankruptcy court|"
    r"court of appeals|before:|summary|opinion|argued|submitted|filed|before|"
    r"circuit judge|circuit judges|district judge|magistrate judge|bankruptcy judge|"
    r"appellant|appellee|plaintiff|defendant|panel|court|district\b|for the\b)\b"
)

# First judge initials in D.C. case numbers to the presiding judge name.
# The corpus uses these suffixes when the full presiding judge line is absent.
_INITIALS_TO_JUDGE = {
    "AB": "Andre Birotte, Jr.",
    "AJB": "Anthony J. Battaglia",
    "AKB": "Amanda K. Brailsford",
    "AMO": "Araceli Martinez-Olguin",
    "AN": "Adrienne C. Nelson",
    "APG": "Andrew P. Gordon",
    "AR": "Karin J. Immergut",
    "ART": "Anne R. Traum",
    "BAS": "Cynthia A. Bashant",
    "BEN": "Roger T. Benitez",
    "BHS": "Benjamin H. Settle",
    "BLF": "Beth Labson Freeman",
    "BMM": "Brian M. Morris",
    "CBM": "Consuelo B. Marshall",
    "CDS": "Cristina D. Silva",
    "CL": "Mark D. Clarke",
    "CRB": "Charles R. Breyer",
    "DAD": "Dale A. Drozd",
    "DCN": "David C. Nye",
    "DDP": "Dean D. Pregerson",
    "DGE": "David G. Estudillo",
    "DLR": "Douglas L. Rayes",
    "DLC": "Dana L. Christensen",
    "DMG": "Dolly M. Gee",
    "DOC": "David O. Carter",
    "DSF": "Dale S. Fischer",
    "DWL": "Dominic Lanza",
    "EFS": "Edward F. Shea",
    "EJD": "Edward J. Davila",
    "EMC": "Edward M. Chen",
    "FMO": "Fernando M. Olguin",
    "GW": "George H. Wu",
    "HSG": "Haywood S. Gilliam, Jr.",
    "HZ": "Marco A. Hernandez",
    "JAD": "Jennifer A. Dorsey",
    "JAM": "John A. Mendez",
    "JAS": "James Alan Soto",
    "JCC": "John C. Coughenour",
    "JCH": "John Charles Hinderaker",
    "JCS": "Joseph C. Spero",
    "JFW": "John F. Walter",
    "JHC": "John H. Chun",
    "JJT": "John Joseph Tuchi",
    "JD": "James Donato",
    "JLB": "John Charles Hinderaker",
    "JLR": "James L. Robart",
    "JLT": "Jennifer L. Thurston",
    "JM": "Jeffrey T. Miller",
    "JMS": "J. Michael Seabright",
    "JO": "Jinsook Ohta",
    "JNW": "Jamal N. Whitehead",
    "JSC": "Jacqueline Scott Corley",
    "JST": "Jon S. Tigar",
    "JSW": "Jeffrey S. White",
    "JVS": "James V. Selna",
    "KJM": "Kimberly J. Mueller",
    "KKE": "Kymberly K. Evanson",
    "LEK": "Leslie E. Kobayashi",
    "LB": "Laurel Beeler",
    "LK": "Lauren J. King",
    "MC": "Michael J. McShane",
    "MCS": "Mark C. Scarsi",
    "MJP": "Marsha J. Pechman",
    "MMD": "Miranda M. Du",
    "IM": "Karin J. Immergut",
    "MO": "Michael W. Mosman",
    "MRA": "Monica Ramirez Almadani",
    "MTL": "Michael T. Liburdi",
    "MWF": "Michael W. Fitzgerald",
    "PA": "Percy Anderson",
    "PSG": "Philip S. Gutierrez",
    "RBM": "Ruth Bermudez Montenegro",
    "RCJ": "Robert Clive Jones",
    "RGK": "R. Gary Klausner",
    "RJB": "Robert J. Bryan",
    "RSH": "Robert Steven Huie",
    "RSL": "Robert S. Lasnik",
    "SAB": "Stanley Allen Bastian",
    "SHR": "Scott H. Rash",
    "SI": "Susan Illston",
    "SLG": "Sharon L. Gleason",
    "SMB": "Susan M. Brnovich",
    "SPG": "Sherilyn Peace Garnett",
    "SPL": "Steven P. Logan",
    "SPW": "Susan P. Watters",
    "SRB": "Susan R. Bolton",
    "SVW": "Stephen V. Wilson",
    "TJH": "Terry J. Hatter, Jr.",
    "TMC": "Tiffany M. Cartwright",
    "TOR": "Thomas O. Rice",
    "TSZ": "Thomas S. Zilly",
    "VC": "Vince Chhabria",
    "WBS": "William B. Shubb",
    "WHA": "William Alsup",
    "WHO": "William H. Orrick",
    "WQH": "William Q. Hayes",
    "YGR": "Yvonne Gonzalez Rogers",
    "DJC": "Daniel J. Calabretta",
}


def _normalize_name(text: str) -> str:
    name = re.sub(r"(?i)^\s*Hon\.?\s*", "", text or "").strip()
    name = name.strip(" \t\r\n,.;:*")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _find_match(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return None
    for pat in (_ROLE_RE, _MAGISTRATE_RE, _BANKRUPTCY_RE):
        match = pat.match(normalized)
        if match:
            return _normalize_name(match.group("name"))
    return None


def _docket_initials(lines: list[dict]) -> str | None:
    for i, item in enumerate(lines[:120]):
        text = " ".join((item.get("text") or "").split())
        if "D.C. No" not in text:
            continue

        candidate_texts = []
        after_label = re.sub(r"(?i)^.*D\.C\.\s+Nos?\.?\s*", "", text).strip()
        if after_label:
            candidate_texts.append(after_label)

        for j in range(i + 1, min(i + 5, len(lines))):
            case_text = " ".join((lines[j].get("text") or "").split())
            if case_text:
                candidate_texts.append(case_text)

        for case_text in candidate_texts:
            m = re.search(r"(?i)\d+:\d+[a-z-]*\d+-?([A-Z]{1,4})(?:-[A-Z]{1,4})?", case_text)
            if not m:
                continue
            first = re.sub(r"[^A-Z]", "", m.group(1).upper())
            if first:
                return first

    return None


def rule_presiding_district_judge(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        def add_name(raw_name: str, source_item: dict) -> None:
            name = _normalize_name(raw_name)
            if not name:
                return
            key = name.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": name}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in source_item:
                    span[field] = source_item[field]
            spans.append(span)

        def is_nameish(text: str) -> bool:
            normalized = re.sub(r"\s+", " ", text or "").strip()
            if not normalized or _NON_NAME_HINT_RE.search(normalized):
                return False
            return bool(_NAMEISH_RE.match(normalized) or _NAME_FRAGMENT_RE.match(normalized))

        def scan_items(items: list[dict]) -> None:
            limit = min(len(items or []), 120)
            for i in range(limit):
                item = items[i] or {}
                text = (item.get("text") or "").strip()
                if not text:
                    continue

                has_role_hint = bool(_ROLE_HINT_RE.search(text))
                has_presiding = "presiding" in text.lower()
                prev_text = ((items[i - 1] or {}).get("text") or "").strip() if i > 0 else ""
                next_text = ((items[i + 1] or {}).get("text") or "").strip() if i + 1 < limit else ""

                direct_name = _find_match(text)
                if direct_name and not (has_role_hint or has_presiding):
                    add_name(direct_name, item)
                    continue

                candidates = []
                if prev_text and is_nameish(prev_text):
                    candidates.append((f"{prev_text} {text}", items[i - 1] or item))
                if next_text and is_nameish(next_text):
                    candidates.append((f"{text} {next_text}", item))
                if prev_text and next_text and is_nameish(prev_text):
                    candidates.append((f"{prev_text} {text} {next_text}", items[i - 1] or item))
                if prev_text and next_text and is_nameish(next_text):
                    candidates.append((f"{prev_text} {text} {next_text}", item))

                matched = False
                for candidate_text, candidate_source in candidates:
                    candidate_name = _find_match(candidate_text)
                    if candidate_name:
                        add_name(candidate_name, candidate_source)
                        matched = True
                        break

                if matched:
                    continue

                if direct_name:
                    add_name(direct_name, item)

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]

        scan_items(lines)
        scan_items(paragraphs)

        if not spans:
            initials = _docket_initials(lines) or _docket_initials(paragraphs)
            if initials and initials in _INITIALS_TO_JUDGE:
                add_name(_INITIALS_TO_JUDGE[initials], {})

        if len(spans) > 1 and not any(
            any(sep in (span.get("text") or "") for sep in (",", " and ", ";"))
            for span in spans
        ):
            aggregate_text = "; ".join(span.get("text") or "" for span in spans if span.get("text"))
            if aggregate_text:
                aggregate_key = aggregate_text.lower()
                if aggregate_key not in seen:
                    seen.add(aggregate_key)
                    aggregate_span = {"text": aggregate_text}
                    first_span = spans[0]
                    for field in ("page_no", "line_no", "paragraph_no"):
                        if field in first_span:
                            aggregate_span[field] = first_span[field]
                    spans.insert(0, aggregate_span)

        if not spans:
            full_text = re.sub(r"\s+", " ", doc.get("text") or "").strip()[:5000]
            if full_text:
                for pat in (_ROLE_RE, _MAGISTRATE_RE, _BANKRUPTCY_RE):
                    match = pat.search(full_text)
                    if match:
                        add_name(match.group("name"), {})
                        break

        return spans
    except Exception:
        return []
