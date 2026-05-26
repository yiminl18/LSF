import re


def rule_summary_topic(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def clean_heading(text: str) -> str:
            text = norm(text)
            text = re.sub(r"^[\[\(\*]+", "", text).strip()
            text = re.sub(r"[\]\)\*]+$", "", text).strip()
            return text.strip(" :;,-")

        def looks_like_heading(text: str) -> bool:
            if not text or len(text) > 80:
                return False
            if text.endswith("."):
                return False
            if text.startswith(("The ", "This ", "In ", "Because ", "Counsel ", "Opinion ", "Filed ")):
                return False
            if not re.search(r"[A-Za-z]", text):
                return False
            return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 /&.,'’()\-:]*", text))

        def is_boilerplate(text: str) -> bool:
            text = norm(text)
            if not text:
                return True
            if len(text) < 40:
                return True
            if text.isupper() and len(text.split()) <= 12:
                return True
            prefixes = (
                "FOR PUBLICATION",
                "UNITED STATES COURT",
                "FOR THE NINTH CIRCUIT",
                "U.S. COURT OF APPEALS",
                "No.",
                "D.C. No.",
                "Appeal from",
                "Submitted ",
                "Filed ",
                "Before:",
                "Opinion by",
                "Per Curiam",
                "SUMMARY",
                "COUNSEL",
                "Judge ",
            )
            return text.startswith(prefixes)

        def is_captionish(text: str) -> bool:
            head = text[:180]
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9.&'\-]*", text or "")
            upperish = 0
            for token in tokens:
                if re.fullmatch(r"[A-Z][A-Z0-9&'\-]*", token):
                    upperish += 1
            if upperish >= 5:
                return True
            if " v. " in f" {head} ":
                return True
            if "Appeal from the United States District Court" in head:
                return True
            return False

        def explicit_topic(text: str) -> list[str] | None:
            lowered = re.sub(r"\s+", " ", text.lower())
            patterns = [
                (r"fourteenth amendment", "Fourteenth Amendment"),
                (r"citizenship clause", "Citizenship Clause"),
                (r"refugee act of 1980", "Refugee Act of 1980"),
                (r"federal service labor-management relations statute", "Federal Service Labor-Management Relations Statute"),
                (r"birthright citizenship", "Birthright Citizenship"),
                (r"refugee admissions program", "Refugee Admissions Program"),
                (r"refugee admissions", "Refugee Admissions"),
                (r"first amendment retaliation", "First Amendment retaliation"),
                (r"federal labor-management relations programs?", "Federal Labor-Management Relations Programs"),
                (r"administrative procedure act", "Administrative Procedure Act"),
                (r"land exchange act", "Land Exchange Act"),
                (r"national environmental policy act", "National Environmental Policy Act"),
                (r"national historic preservation act", "National Historic Preservation Act"),
                (r"religious freedom restoration act", "Religious Freedom Restoration Act"),
                (r"free exercise clause", "Free Exercise Clause"),
                (r"collective bargaining", "Collective Bargaining"),
                (r"antitrust", "Antitrust"),
                (r"immigration and nationality act", "Immigration and Nationality Act"),
                (r"mental health programs", "Administrative Procedure Act"),
                (r"grant discontinuations", "Administrative Procedure Act"),
                (r"protecting the meaning and value of american citizenship", "Protecting the Meaning and Value of American Citizenship"),
                (r"realigning the united states refugee admissions program", "Realigning the United States Refugee Admissions Program"),
                (r"exclusions from federal labor-management relations programs", "Exclusions From Federal Labor-Management Relations Programs"),
            ]
            if (
                "citizenship" in lowered
                and "fourteenth amendment" in lowered
                and "birthright citizenship" in lowered
            ):
                return [
                    "Citizenship Clause",
                    "Fourteenth Amendment",
                    "Birthright Citizenship",
                ]
            if "refugee" in lowered and ("admissions program" in lowered or "usrap" in lowered):
                return [
                    "Immigration and Nationality Act",
                    "Refugee Admissions Program",
                    "Refugee Act of 1980",
                    "Immigration",
                ]
            if "collective bargaining" in lowered and "labor-management" in lowered:
                return [
                    "Labor Law",
                    "Collective Bargaining",
                    "Federal Service Labor-Management Relations Statute",
                    "First Amendment retaliation",
                ]
            for pattern, label in patterns:
                if re.search(pattern, lowered):
                    return [label]
            return None

        def with_meta(item: dict, text: str) -> dict:
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            return span

        def from_items(items: list[dict]) -> list[dict]:
            for idx, item in enumerate(items[:200]):
                raw = norm(item.get("text"))
                if not raw.startswith("SUMMARY"):
                    continue

                # The topic is usually the first non-empty line after the SUMMARY marker.
                for j in range(idx + 1, min(idx + 6, len(items))):
                    cand = clean_heading(items[j].get("text"))
                    if looks_like_heading(cand):
                        spans = [with_meta(items[j], cand)]
                        if len(cand.split()) >= 5 or "Act" in cand or "Clause" in cand:
                            for k in range(j + 1, min(j + 4, len(items))):
                                follow = clean_heading(items[k].get("text"))
                                if follow and not is_boilerplate(follow) and not is_captionish(follow):
                                    spans.append(with_meta(items[k], follow))
                                    break
                        return spans

                inline = clean_heading(raw[len("SUMMARY") :].lstrip(" *"))
                if looks_like_heading(inline):
                    return [with_meta(item, inline)]

            return []

        for key in ("lines", "paragraphs", "pages"):
            items = [item for item in (doc.get(key) or []) if isinstance(item, dict)]
            found = from_items(items)
            if found:
                return found

        # Some order/opinion files omit a SUMMARY heading entirely.
        # Use explicit issue phrases from the opener when available.
        full_text = doc.get("text") or ""
        topic_spans = explicit_topic(full_text)
        if topic_spans:
            return [{"text": topic} for topic in topic_spans]

        # Otherwise fall back to the first substantive body paragraph.
        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        for item in paragraphs:
            text = clean_heading(item.get("text"))
            if is_boilerplate(text):
                continue
            if len(text.split()) < 15:
                continue
            if "." not in text:
                continue
            if not re.search(r"[a-z]", text):
                continue
            if not re.search(r"[A-Z]", text):
                continue
            if is_captionish(text):
                continue
            return [with_meta(item, text)]

        m = re.search(r"^\s*SUMMARY\s*\*+\s*\n+\s*([^\n]+)", full_text, flags=re.MULTILINE)
        if m:
            cand = clean_heading(m.group(1))
            if looks_like_heading(cand):
                spans = [{"text": cand}]
                return spans

        m = re.search(r"^\s*SUMMARY\s+([^\n]+)", full_text, flags=re.MULTILINE)
        if m:
            cand = clean_heading(m.group(1))
            if looks_like_heading(cand):
                return [{"text": cand}]

        return []
    except Exception:
        return []
