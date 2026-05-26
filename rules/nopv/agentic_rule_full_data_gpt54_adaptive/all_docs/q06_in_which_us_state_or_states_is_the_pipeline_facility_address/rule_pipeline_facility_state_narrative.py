def rule_pipeline_facility_state_narrative(doc: dict) -> list[dict]:
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
        inspection_pat = re.compile(
            r"(?i)\b(?:inspected|inspection of|conducted an inspection|performed an on-site "
            r"investigation|conducted an investigation|investigated|reviewed|review of|"
            r"virtually inspected|observed excavation|acting as agents of PHMSA)\b"
        )
        subject_pat = re.compile(
            r"(?i)\b(?:pipeline|pipelines|pipeline system|pipeline systems|facility|facilities|"
            r"control room|distribution system|distribution systems|drug and alcohol|"
            r"integrity management|records|procedures|operations|maintenance|program|"
            r"station|compressor station|compressor|terminal|plant|storage facility|"
            r"storage facilities|underground natural gas storage|UNGSF|segment|lateral|"
            r"line [A-Z0-9#\\-]+|construction project|construction records|incident|release|"
            r"assets|annual report records)\b"
        )
        direct_location_pat = re.compile(
            r"(?i)\b(?:located in|located at|facility in|facilities in|facility located in|"
            r"pipeline system located in|pipeline systems located in|pipeline system in|"
            r"pipeline systems in|distribution system in|distribution systems in|"
            r"control room in|terminal in|plant in|station in|storage facility in|"
            r"storage facilities in|underground natural gas storage facilities in|"
            r"records and facilities|records, procedures and facilities|near [A-Z][A-Za-z .'-]+,\s*"
            r"(?:%s|%s)|at [0-9A-Za-z .#&'/-]+,\s*[A-Z][A-Za-z .'-]+,\s*(?:%s|%s))\b"
            % (state_name_pat, state_abbrev_pat, state_name_pat, state_abbrev_pat)
        )
        stop_pat = re.compile(
            r"(?i)\b(?:As a result of the inspection|As a result of the investigation|"
            r"Based on the inspection|The items? inspected|The items? investigated|"
            r"The item inspected|The item investigated)\b"
        )

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def has_state(text: str) -> bool:
            return bool(
                state_name_re.search(text or "")
                or state_abbrev_re.search(text or "")
                or extra_location_re.search(text or "")
            )

        def trim(text: str) -> str:
            text = norm(text)
            if not text:
                return ""
            dear_m = re.search(r"Dear [^:]{0,120}:\s*", text[:500])
            if dear_m:
                text = text[dear_m.end() :].strip()
            m = stop_pat.search(text)
            if m and m.start() > 0:
                text = text[: m.start()].rstrip(" ;,")
            text = re.sub(r",?\s+and the Gulf of Mexico\b", "", text, flags=re.I)
            text = re.sub(r",?\s+in the Gulf of Mexico\b", "", text, flags=re.I)
            return text

        def sentence_split(text: str) -> list[str]:
            parts = [norm(part) for part in re.split(r"(?<=[.!?])\s+", norm(text))]
            return [part for part in parts if part]

        def paragraph_score(text: str) -> tuple[int, str]:
            text = trim(text)
            if not text or not has_state(text):
                return (0, "")

            sents = sentence_split(text)
            if not sents:
                sents = [text]

            matched = []
            for sent in sents:
                if not has_state(sent):
                    continue
                if direct_location_pat.search(sent):
                    matched.append(sent)
                    continue
                if inspection_pat.search(sent) and subject_pat.search(sent):
                    matched.append(sent)
                    continue
                if subject_pat.search(sent) and re.search(r"(?i)\b(?:in|near|at|throughout|across)\b", sent):
                    matched.append(sent)

            if matched:
                snippet = " ".join(matched)
                score = 3 if any(direct_location_pat.search(sent) for sent in matched) else 2
                return (score, snippet)

            if direct_location_pat.search(text):
                return (2, text)
            if inspection_pat.search(text) and subject_pat.search(text):
                return (1, text)
            return (0, "")

        def add(out: list[dict], seen: set[tuple], text: str, page_no=None, paragraph_no=None, line_no=None):
            text = norm(text)
            if not text:
                return
            key = (text.lower(), page_no, paragraph_no, line_no)
            if key in seen:
                return
            seen.add(key)
            span = {"text": text}
            if page_no is not None:
                span["page_no"] = page_no
            if paragraph_no is not None:
                span["paragraph_no"] = paragraph_no
            if line_no is not None:
                span["line_no"] = line_no
            out.append(span)

        out: list[dict] = []
        seen: set[tuple] = set()

        for item in (doc.get("paragraphs") or [])[:12]:
            trimmed = trim(item.get("text", ""))
            if not trimmed or not has_state(trimmed):
                continue
            if not (inspection_pat.search(trimmed) or direct_location_pat.search(trimmed)):
                continue
            add(out, seen, trimmed, page_no=item.get("page_no"), paragraph_no=item.get("paragraph_no"))
            return out[:1]

        best = None
        for item in (doc.get("paragraphs") or [])[:18]:
            score, snippet = paragraph_score(item.get("text", ""))
            if score <= 0:
                continue
            candidate = (
                score,
                item.get("page_no", 10**9),
                item.get("paragraph_no", 10**9),
                snippet,
                item.get("page_no"),
                item.get("paragraph_no"),
            )
            if best is None or score > best[0] or (score == best[0] and candidate[1:3] < best[1:3]):
                best = candidate
                if score >= 3:
                    break

        if best is not None:
            add(out, seen, best[3], page_no=best[4], paragraph_no=best[5])
            return out[:1]

        lines = [item for item in (doc.get("lines") or [])[:80] if item.get("text", "").strip()]
        dear_idx = None
        for idx, item in enumerate(lines):
            if "Dear " in item.get("text", ""):
                dear_idx = idx
                break
        if dear_idx is not None:
            intro_lines = []
            for item in lines[dear_idx + 1 : dear_idx + 12]:
                text = norm(item.get("text", ""))
                if not text:
                    continue
                if re.search(r"(?i)^As a result of the inspection|^As a result of the investigation|^Based on the inspection", text):
                    break
                intro_lines.append(item)
            if intro_lines:
                merged = norm(" ".join(item["text"] for item in intro_lines))
                score, snippet = paragraph_score(merged)
                if score > 0:
                    add(out, seen, snippet, page_no=intro_lines[0].get("page_no"), line_no=intro_lines[0].get("line_no"))
                    return out[:1]

        return out[:1]
    except Exception:
        return []
