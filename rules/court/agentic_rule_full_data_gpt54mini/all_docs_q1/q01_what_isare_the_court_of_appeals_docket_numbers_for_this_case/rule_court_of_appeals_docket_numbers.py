def rule_court_of_appeals_docket_numbers(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        spans = []
        seen = set()

        court_heading_re = re.compile(
            r"(UNITED STATES COURT OF APPEALS|U\.S\. COURT OF APPEALS|FOR THE NINTH CIRCUIT|NINTH CIRCUIT)",
            re.IGNORECASE,
        )
        docket_re = re.compile(r"\b\d{2,}-\d+\b")
        nos_only_re = re.compile(r"^\s*Nos?\.?\s*$", re.IGNORECASE)
        bare_docket_re = re.compile(r"^\s*\d{2,}-\d+(?:\s*,\s*\d{2,}-\d+)*\s*$")

        in_header = False
        in_nos_block = False
        body_started = False

        for line in lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue

            upper = text.upper()
            if not in_header and court_heading_re.search(text):
                in_header = True

            if body_started:
                continue

            if in_header and (
                upper.startswith("SUMMARY")
                or text.startswith("Before:")
                or text.startswith("This summary")
            ):
                body_started = True
                continue

            if not in_header:
                continue

            if nos_only_re.match(text):
                in_nos_block = True
                continue

            if text.lower().startswith("no.") or text.lower().startswith("nos."):
                nums = docket_re.findall(text)
                if nums:
                    added = False
                    for num in nums:
                        if num not in seen:
                            seen.add(num)
                            added = True
                    if added:
                        spans.append(
                            {
                                "text": text,
                                "page_no": line.get("page_no"),
                                "line_no": line.get("line_no"),
                            }
                        )
                in_nos_block = text.lower().startswith("nos.")
                continue

            if in_nos_block and bare_docket_re.match(text):
                nums = docket_re.findall(text)
                if nums:
                    added = False
                    for num in nums:
                        if num not in seen:
                            seen.add(num)
                            added = True
                    if added:
                        spans.append(
                            {
                                "text": text,
                                "page_no": line.get("page_no"),
                                "line_no": line.get("line_no"),
                            }
                        )
                    continue

            if in_nos_block:
                in_nos_block = False

        return spans
    except Exception:
        return []
