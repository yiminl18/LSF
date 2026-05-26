import re


def rule_proposed_order_operator(doc: dict) -> list[dict]:
    try:
        legalish_re = re.compile(
            r"(?i)\b(?:llc|l\.p\.|lp\b|inc\.?|co\.?|company|corporation|corp\.?|"
            r"transmission|pipeline|utility|utilities|operating|products|storage|"
            r"midstream|holdco|holdings|partners|services|energy|gas|chemical|chemicals)\b"
        )
        patterns = [
            re.compile(r"proposes to issue to\s+(.+?)\s+a\s+(?:Compliance Order|civil penalty)\b", re.I),
            re.compile(r"proposes to issue a Compliance Order to\s+(.+?)(?:[.;]|$)", re.I),
        ]

        for para in doc.get("paragraphs", []):
            text = " ".join(((para.get("text") or "").strip()).split())
            if "proposes to issue" not in text.lower():
                continue
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                entity = re.sub(r"\s+\([^)]*\)", "", match.group(1)).strip(" ,;.")
                if not entity:
                    continue
                if not legalish_re.search(entity):
                    continue
                return [
                    {
                        "text": entity,
                        "page_no": para.get("page_no"),
                        "paragraph_no": para.get("paragraph_no"),
                    }
                ]
        return []
    except Exception:
        return []
