import re


def rule_principal_executive_offices_cover_address(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()

        def make_span(source: dict) -> dict:
            span = {"text": norm(source.get("text", ""))}
            if source.get("page_no") is not None:
                span["page_no"] = source["page_no"]
            if source.get("line_no") is not None:
                span["line_no"] = source["line_no"]
            return span

        label_re = re.compile(r"principal\s+executive\s+offices", re.IGNORECASE)
        phone_re = re.compile(
            r"(\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b)"
        )
        stop_re = re.compile(
            r"(state or other jurisdiction|irs employer identification|registrant[’']?s telephone number|"
            r"telephone number|securities registered pursuant|indicate by check mark|table of contents|"
            r"commission file number|form 10-k|form 10-q|exact name of registrant|former name or former address)",
            re.IGNORECASE,
        )
        street_re = re.compile(
            r"\b(street|st\.|road|rd\.|avenue|ave\.|drive|dr\.|boulevard|blvd\.|plaza|lane|ln\.|"
            r"way|highway|hwy\.|court|ct\.|parkway|pkwy\.|circle|cir\.|suite|ste\.|floor|building|"
            r"bldg\.|tower|center|centre)\b",
            re.IGNORECASE,
        )
        zip_re = re.compile(r"\b\d{5}(?:-\d{4})?\b")
        us_state_re = re.compile(
            r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|"
            r"MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\b"
        )
        country_re = re.compile(r"\b(united states|united kingdom|canada|switzerland|japan|mexico|ireland|australia)\b", re.IGNORECASE)

        def looks_like_address(text: str) -> bool:
            low = norm(text).lower()
            if not low:
                return False
            if phone_re.search(low):
                return False
            if label_re.search(low):
                return False
            if stop_re.search(low):
                return False
            if re.fullmatch(r"\d{5}(?:-\d{4})?", low):
                return False
            if zip_re.search(low) and any(ch.isalpha() for ch in low):
                return True
            if country_re.search(low):
                return True
            if street_re.search(low):
                return True
            if us_state_re.search(text) and any(ch.isdigit() for ch in text):
                return True
            if re.search(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*,\s*[A-Z][A-Za-z.'-]+", text):
                return True
            if re.search(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*\s+[A-Z]{2}\b", text):
                return True
            if "," in text and any(ch.isdigit() for ch in text):
                return True
            return False

        label_idx = None
        for idx, line in enumerate(lines):
            if line.get("page_no") not in (1, 2):
                continue
            if label_re.search(norm(line.get("text", ""))):
                label_idx = idx
                break

        if label_idx is None:
            return []

        selected: list[dict] = []
        for idx in range(label_idx - 1, max(-1, label_idx - 10), -1):
            raw = norm(lines[idx].get("text", ""))
            if not raw:
                continue
            low = raw.lower()
            if stop_re.search(low):
                if selected:
                    break
                continue
            if phone_re.search(raw):
                continue
            if looks_like_address(raw):
                selected.append(lines[idx])
                continue
            if selected:
                break

        selected.reverse()

        out: list[dict] = []
        seen: set[str] = set()
        for source in selected:
            cleaned = norm(source.get("text", ""))
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(make_span(source))
        return out
    except Exception:
        return []
