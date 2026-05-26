def rule_pipeline_facility_state_address_fallback(doc: dict) -> list[dict]:
    try:
        import re

        state_name_pat = (
            r"Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
            r"Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|"
            r"Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|"
            r"Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|"
            r"New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
            r"Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|"
            r"Virginia|Washington|West Virginia|Wisconsin|Wyoming"
        )
        state_abbrev_pat = (
            r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|"
            r"MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
            r"VT|VA|WA|WV|WI|WY"
        )
        extra_location_pat = (
            r"Puerto Rico|Alberta|British Columbia|Manitoba|New Brunswick|"
            r"Newfoundland and Labrador|Nova Scotia|Ontario|Quebec|Saskatchewan"
        )
        state_name_re = re.compile(r"\b(?:%s)\b" % state_name_pat, re.I)
        state_abbrev_re = re.compile(r"\b(?:%s)\b" % state_abbrev_pat)
        extra_location_re = re.compile(r"\b(?:%s)\b" % extra_location_pat, re.I)
        zip_line_pat = re.compile(
            r"\b(?:%s|%s|%s)\b[\s,]+(?:\d{5}(?:-\d{4})?)\b" % (state_name_pat, state_abbrev_pat, extra_location_pat)
        )
        narrative_pat = re.compile(
            r"(?i)\b(?:inspected|inspection of|conducted an inspection|"
            r"performed an on-site investigation|conducted an investigation|"
            r"investigated|reviewed|review of|virtually inspected|located in|"
            r"located at|release occurred|incident near)\b"
        )
        legal_entity_pat = re.compile(
            r"(?i)\b(?:Inc\.?|LLC|L\.P\.|LP|Company|Corporation|Corp\.?|Pipeline|"
            r"Transmission|Storage|Partners|Department|Utilities|Energy|Gas|Water|Power)\b"
        )

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def has_state(text: str) -> bool:
            return bool(
                state_name_re.search(text or "")
                or state_abbrev_re.search(text or "")
                or extra_location_re.search(text or "")
            )

        def add(out: list[dict], seen: set[tuple], text: str, page_no=None, line_no=None):
            text = norm(text)
            if not text:
                return
            key = (text.lower(), page_no, line_no)
            if key in seen:
                return
            seen.add(key)
            span = {"text": text}
            if page_no is not None:
                span["page_no"] = page_no
            if line_no is not None:
                span["line_no"] = line_no
            out.append(span)

        for item in (doc.get("paragraphs") or [])[:30]:
            text = norm(item.get("text", ""))
            if has_state(text) and narrative_pat.search(text):
                return []

        lines = [item for item in (doc.get("lines") or [])[:90] if item.get("text", "").strip()]
        out: list[dict] = []
        seen: set[tuple] = set()

        head_text = norm(" ".join(item["text"] for item in lines[:25]))
        if re.search(r"(?i)\boffshore\b", head_text):
            return []

        cpf_idx = None
        for idx, item in enumerate(lines):
            if "CPF" in item.get("text", ""):
                cpf_idx = idx
                break

        if cpf_idx is not None:
            start = max(0, cpf_idx - 8)
            for idx in range(cpf_idx - 1, start - 1, -1):
                line_text = norm(lines[idx]["text"])
                if not zip_line_pat.search(line_text):
                    continue
                block_start = max(start, idx - 2)
                block_lines = [norm(lines[j]["text"]) for j in range(block_start, idx + 1)]
                snippet = " ".join(part for part in block_lines if part)
                if has_state(snippet):
                    add(out, seen, snippet, page_no=lines[block_start].get("page_no"), line_no=lines[block_start].get("line_no"))
                    return out[:1]

        for item in (doc.get("paragraphs") or [])[:15]:
            text = norm(item.get("text", ""))
            if not has_state(text):
                continue
            if re.search(r"(?i)\bproposes to issue to\b", text) and legal_entity_pat.search(text):
                add(out, seen, text, page_no=item.get("page_no"), line_no=None)
                return out[:1]

        for idx, item in enumerate(lines[:35]):
            text = norm(item.get("text", ""))
            if not has_state(text):
                continue
            if zip_line_pat.search(text):
                block_start = max(0, idx - 2)
                block_lines = [norm(lines[j]["text"]) for j in range(block_start, idx + 1)]
                snippet = " ".join(part for part in block_lines if part)
                if legal_entity_pat.search(snippet):
                    add(out, seen, snippet, page_no=lines[block_start].get("page_no"), line_no=lines[block_start].get("line_no"))
                    return out[:1]
            if legal_entity_pat.search(text):
                add(out, seen, text, page_no=item.get("page_no"), line_no=item.get("line_no"))
                return out[:1]

        for item in lines[:120]:
            text = norm(item.get("text", ""))
            if has_state(text) and legal_entity_pat.search(text):
                add(out, seen, text, page_no=item.get("page_no"), line_no=item.get("line_no"))
                return out[:1]

        return out[:1]
    except Exception:
        return []
