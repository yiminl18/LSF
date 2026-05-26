def rule_incorporation_ein_cover_page(doc: dict) -> list[dict]:
    try:
        import re

        def norm(text: str) -> str:
            return " ".join(str(text or "").split())

        state_label_pat = re.compile(
            r"(?i)\b(?:state or other jurisdiction of(?: incorporation(?: or organization)?)?"
            r"|jurisdiction of incorporation(?: or organization)?"
            r"|incorporation or organization)\b"
        )
        ein_label_pat = re.compile(
            r"(?i)\b(?:irs employer identification no\.?|"
            r"i\.r\.s\. employer identification no\.?|"
            r"employer identification no\.?)\b"
        )
        ein_value_pat = re.compile(r"\b\d{2}-\d{7}\b")
        state_value_pat = re.compile(
            r"(?i)\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
            r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|"
            r"Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|"
            r"Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
            r"North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
            r"South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
            r"West Virginia|Wisconsin|Wyoming|District of Columbia|Puerto Rico|Guam|"
            r"U\.S\. Virgin Islands|American Samoa|Jersey|Bermuda|Cayman Islands|"
            r"British Columbia|Ontario|Quebec|Singapore|Hong Kong|Ireland|Luxembourg|"
            r"Switzerland|Netherlands|England and Wales|Australia|New South Wales|Victoria)\b"
        )

        def score(snippet: str) -> int:
            low = snippet.lower()
            val = 0
            if state_label_pat.search(snippet):
                val += 6
            if ein_label_pat.search(snippet):
                val += 6
            if ein_value_pat.search(snippet):
                val += 5
            if state_value_pat.search(snippet):
                val += 2
            if "exact name of registrant" in low:
                val += 1
            if "address of principal executive offices" in low:
                val -= 1
            if "table of contents" in low:
                val -= 2
            return val

        def collect_windows(items: list[dict]) -> list[tuple[int, int, int | None, int | None, str]]:
            out: list[tuple[int, int, int | None, int | None, str]] = []
            if not items:
                return out
            limit = min(len(items), 120)
            for idx in range(limit):
                item = items[idx]
                text = norm(item.get("text"))
                if not text:
                    continue
                low = text.lower()
                if not (
                    state_label_pat.search(text)
                    or ein_label_pat.search(text)
                    or ein_value_pat.search(text)
                    or "incorporat" in low
                    or "jurisdiction" in low
                    or "employer identification" in low
                ):
                    continue
                start = max(0, idx - 3)
                end = min(limit, idx + 4)
                snippet_lines = [norm(items[j].get("text")) for j in range(start, end)]
                snippet_lines = [line for line in snippet_lines if line]
                if not snippet_lines:
                    continue
                snippet = "\n".join(snippet_lines).strip()
                if not snippet:
                    continue
                out.append(
                    (
                        score(snippet),
                        idx,
                        item.get("page_no"),
                        item.get("line_no"),
                        snippet,
                    )
                )
            return out

        ordered_lines = sorted(
            doc.get("lines") or [],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        candidates = collect_windows(ordered_lines)

        if not candidates and doc.get("pages"):
            pages = sorted(doc.get("pages") or [], key=lambda x: x.get("page_no", 10**9))
            if pages:
                first_page_text = norm(pages[0].get("text"))
                if first_page_text:
                    m = re.search(
                        r"(?is)(.{0,120}?"
                        r"(?:state or other jurisdiction of(?: incorporation(?: or organization)?)?|"
                        r"jurisdiction of incorporation(?: or organization)?|"
                        r"incorporation or organization|"
                        r"irs employer identification no\.?|"
                        r"i\.r\.s\. employer identification no\.?|"
                        r"employer identification no\.?)"
                        r".{0,200})",
                        first_page_text,
                    )
                    if m:
                        snippet = norm(m.group(1))
                        candidates.append((score(snippet), 0, pages[0].get("page_no"), None, snippet))

        if not candidates:
            full_text = norm(doc.get("text"))
            if full_text:
                m = re.search(
                    r"(?is)(.{0,120}?"
                    r"(?:state or other jurisdiction of(?: incorporation(?: or organization)?)?|"
                    r"jurisdiction of incorporation(?: or organization)?|"
                    r"incorporation or organization|"
                    r"irs employer identification no\.?|"
                    r"i\.r\.s\. employer identification no\.?|"
                    r"employer identification no\.?)"
                    r".{0,220})",
                    full_text,
                )
                if m:
                    snippet = norm(m.group(1))
                    candidates.append((score(snippet), 0, None, None, snippet))

        if not candidates:
            return []

        candidates.sort(key=lambda item: (-item[0], item[1]))
        _, _, page_no, line_no, snippet = candidates[0]
        span = {"text": snippet}
        if page_no is not None:
            span["page_no"] = page_no
        if line_no is not None:
            span["line_no"] = line_no
        return [span]
    except Exception:
        return []
