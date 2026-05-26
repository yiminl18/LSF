def rule_pipeline_facility_state_intro(doc: dict) -> list[dict]:
    try:
        import re

        state_name_pat = (
            r"Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
            r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Iowa|Kansas|"
            r"Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|"
            r"Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
            r"New Hampshire|New Jersey|New Mexico|New York|North Carolina|"
            r"North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
            r"South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|"
            r"Virginia|Washington|West Virginia|Wisconsin|Wyoming"
        )
        state_abbrev_pat = (
            r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IA|KS|KY|LA|MD|MA|MI|"
            r"MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|PA|RI|SC|SD|TN|TX|UT|"
            r"VT|VA|WA|WV|WI|WY"
        )
        state_pat = re.compile(r"\b(?:%s|%s)\b" % (state_name_pat, state_abbrev_pat), re.I)
        intro_pat = re.compile(
            r"(?i)\b(?:located in|located at|facility in|facility located in|"
            r"facilities in|pipeline system in|pipeline systems in|"
            r"pipeline system located in|pipeline systems located in|"
            r"distribution system in|distribution systems in|control room in|"
            r"control rooms in|LNG facility in|tank farm in|terminal in|"
            r"plant in|station in|inspected(?: your)?(?: [^.\n]{0,80})?\bin)\b"
        )
        context_pat = re.compile(
            r"(?i)\b(?:facility|facilities|distribution|control room|control rooms|"
            r"terminal|plant|station|tank farm|lng|operations and maintenance|o&m|"
            r"records and facilities|interconnect|lateral|compressor|assets|well|"
            r"pipeline facility|pipeline facilities|pipeline system|pipeline systems|"
            r"pipelines|distribution system|distribution systems|inspection|"
            r"inspection system|inspection systems)\b"
        )

        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def split_fragments(text: str, preserve_newlines: bool = False) -> list[str]:
            fragments = []
            if preserve_newlines:
                blocks = [norm(text)]
            else:
                blocks = re.split(r"\n+", text or "")
            for block in blocks:
                for frag in re.split(r"(?<=[.!?])\s+", block):
                    frag = norm(frag)
                    if frag:
                        fragments.append(frag)
            return fragments

        def best_snippet(text: str) -> str:
            low = text.lower()
            anchors = [
                "facility located in",
                "pipeline system located in",
                "pipeline systems located in",
                "distribution system in",
                "distribution systems in",
                "control rooms in",
                "control room in",
                "lng facility in",
                "tank farm facility in",
                "facility in",
                "located in",
                "located at",
                "terminal in",
                "plant in",
                "station in",
                "inspected",
            ]
            starts = [low.find(a) for a in anchors if low.find(a) != -1]
            if starts:
                start = min(starts)
                tail = text[start:]
                end_m = re.search(r"[.!?](?:\s|$)", tail)
                end = start + end_m.end() if end_m else len(text)
                snippet = norm(text[start:end])
                if state_pat.search(snippet):
                    snippet = re.sub(r",?\s+and the Gulf of Mexico\b.*$", "", snippet, flags=re.I).rstrip(",; ")
                    return snippet
            m = state_pat.search(text)
            if m:
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 120)
                snippet = norm(text[start:end])
                return re.sub(r",?\s+and the Gulf of Mexico\b.*$", "", snippet, flags=re.I).rstrip(",; ")
            return ""

        sources = []
        for item in (doc.get("paragraphs") or [])[:30]:
            if item.get("text"):
                sources.append(("paragraph", item.get("page_no"), item.get("paragraph_no"), None, item["text"]))
        for item in (doc.get("lines") or [])[:120]:
            if item.get("text"):
                sources.append(("line", item.get("page_no"), None, item.get("line_no"), item["text"]))

        seen = set()
        out = []

        def add(text, page_no=None, paragraph_no=None, line_no=None):
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

        for kind, page_no, paragraph_no, line_no, text in sources:
            for frag in split_fragments(text, preserve_newlines=(kind == "paragraph")):
                low = frag.lower()
                if not state_pat.search(frag):
                    continue
                if intro_pat.search(frag) or (
                    ("inspected" in low or "located" in low or "facility" in low or "facilities" in low or "distribution" in low or "across" in low)
                    and context_pat.search(frag)
                ):
                    snippet = best_snippet(frag)
                    if snippet:
                        add(snippet, page_no, paragraph_no, line_no)
                if len(out) >= 4:
                    break
            if len(out) >= 4:
                break

        if not out:
            full = doc.get("text") or ""
            m = state_pat.search(full)
            if m and intro_pat.search(full):
                start = max(0, m.start() - 120)
                end = min(len(full), m.end() + 160)
                add(best_snippet(full[start:end]) or full[start:end])

        return out[:1]
    except Exception:
        return []
