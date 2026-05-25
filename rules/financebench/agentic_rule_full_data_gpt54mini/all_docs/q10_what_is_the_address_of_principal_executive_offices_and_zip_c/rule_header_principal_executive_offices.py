import re


def rule_header_principal_executive_offices(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        normalized = []
        for idx, line in enumerate(lines):
            text = line.get("text") or ""
            match_text = (
                text.replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u00a0", " ")
                .replace("\u202f", " ")
                .replace("\u2009", " ")
            )
            match_text = re.sub(r"\s+", " ", match_text).strip()
            normalized.append(
                {
                    "idx": idx,
                    "page_no": line.get("page_no"),
                    "line_no": line.get("line_no"),
                    "text": text,
                    "lower": match_text.lower(),
                }
            )

        pages = [item["page_no"] for item in normalized if item["page_no"] is not None]
        if not pages:
            return []
        first_page = min(pages)
        candidate_pages = {first_page, first_page + 1}

        label_patterns = (
            "address of principal executive offices",
            "address and telephone number, including area code, of registrant's principal executive offices",
            "address and telephone number, including area code, of registrant’s principal executive offices",
            "address of principal executive offices and zip code",
            "address of principal executive offices, including zip code",
            "address of principal executive offices) (zip code",
            "address of principal executive offices) (zip code)",
            "address of principal executive offices) (zipcode",
            "address of principal executive offices and zip code",
            "address of principal executive offices and zip code)",
            "address of principal executive offices and zip code ",
            "address of principal executive offices) (zip code ",
        )

        block_keywords = (
            "telephone",
            "employer identification",
            "identification no",
            "commission file",
            "exact name of registrant",
            "securities registered",
            "trading symbol",
            "incorporation or organization",
            "state or other jurisdiction",
            "former name",
            "table of contents",
            "annual report",
            "current report",
            "indicate by check mark",
            "registrant's telephone number",
            "registrant’s telephone number",
            "principal executive officer",
        )
        street_keywords = (
            "street",
            "st.",
            "st ",
            "road",
            "rd.",
            "rd ",
            "avenue",
            "ave.",
            "ave ",
            "boulevard",
            "blvd.",
            "blvd ",
            "drive",
            "dr.",
            "dr ",
            "lane",
            "ln.",
            "ln ",
            "way",
            "court",
            "ct.",
            "ct ",
            "circle",
            "cir.",
            "cir ",
            "highway",
            "hwy.",
            "hwy ",
            "park",
            "parkway",
            "pkwy",
            "suite",
            "ste.",
            "ste ",
            "floor",
            "p.o. box",
            "po box",
            "box ",
        )

        phone_re = re.compile(r"\(?\d{3}\)?[\s.-]*\d{3}[\s.-]*\d{4}")
        ein_re = re.compile(r"\d{2}-\d{7}")
        zip_re = re.compile(r"\d{5}(?:-\d{4})?")

        def is_blocked_text(text: str, low: str) -> bool:
            if any(pat in low for pat in block_keywords):
                return True
            if ein_re.fullmatch(text):
                return True
            if phone_re.search(text):
                return True
            return False

        def is_address_line(text: str, low: str) -> bool:
            if zip_re.fullmatch(text):
                return True
            if any(pat in low for pat in street_keywords):
                return True
            if any(ch.isdigit() for ch in text):
                return True
            if "," in text and len(text) <= 60:
                return True
            if re.fullmatch(r"[A-Z]{2}", text.strip()):
                return True
            return False

        spans = []
        for i, item in enumerate(normalized):
            if item["page_no"] not in candidate_pages:
                continue
            lower = item["lower"]
            if not lower:
                continue
            if not any(pat in lower for pat in label_patterns):
                continue

            collected = []
            start = max(0, i - 5)
            end = min(len(normalized), i + 5)
            for j in range(i - 1, start - 1, -1):
                cur = normalized[j]
                if cur["page_no"] != item["page_no"]:
                    continue
                text = cur["text"].strip()
                low = cur["lower"]
                if not text:
                    continue
                if any(pat in low for pat in label_patterns):
                    break
                if is_blocked_text(text, low):
                    if collected:
                        break
                    continue
                if is_address_line(text, low):
                    collected.append(cur)
                elif collected:
                    break

            collected.reverse()

            has_zip = any(zip_re.fullmatch(x["text"].strip()) for x in collected)
            if not has_zip:
                for j in range(i + 1, min(len(normalized), i + 5)):
                    cur = normalized[j]
                    if cur["page_no"] != item["page_no"]:
                        continue
                    text = cur["text"].strip()
                    low = cur["lower"]
                    if not text:
                        continue
                    if "zip code" in low:
                        # Include lines immediately before a standalone ZIP label.
                        for k in range(j - 1, max(i, j - 3), -1):
                            prev = normalized[k]
                            if prev["page_no"] != item["page_no"]:
                                continue
                            prev_text = prev["text"].strip()
                            prev_low = prev["lower"]
                            if not prev_text:
                                continue
                            if is_blocked_text(prev_text, prev_low):
                                continue
                            if is_address_line(prev_text, prev_low):
                                collected.append(prev)
                        collected.sort(key=lambda x: x["idx"])
                        has_zip = any(zip_re.fullmatch(x["text"].strip()) for x in collected)
                        break

            if not collected:
                continue

            collected.sort(key=lambda x: x["idx"])
            joined = " ".join(x["text"].strip() for x in collected if x["text"].strip())
            joined_low = joined.lower()
            if len(joined.split()) < 3:
                continue
            if not any(pat in joined_low for pat in street_keywords) and "," not in joined:
                continue
            if not any(ch.isdigit() for ch in joined):
                continue
            first = collected[0]
            last = collected[-1]
            spans.append(
                {
                    "text": joined,
                    "page_no": first["page_no"],
                    "line_no": first["line_no"],
                    "end_line_no": last["line_no"],
                }
            )
            break

        return spans
    except Exception:
        return []
