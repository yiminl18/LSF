import re


_CPF_RE = re.compile(
    r"\bCPF\s*:?\s*([1-5])\s*-\s*(20\d{2})\s*-\s*(\d{3,4})\s*(?:-\s*|\s+)NOPV\b",
    re.IGNORECASE,
)
_DOC_NAME_RE = re.compile(
    r"(?<!\d)([1-5])(20\d{2})(\d{3})NOPV(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def rule_cpf_case_number_assigned_to_enforcement_action(doc: dict) -> list[dict]:
    try:
        for source_key in ("lines", "paragraphs"):
            for item in doc.get(source_key) or []:
                text = (item.get("text") or "").strip()
                if not text or "CPF" not in text.upper():
                    continue

                match = _CPF_RE.search(text)
                if not match:
                    continue

                normalized = f"CPF {match.group(1)}-{match.group(2)}-{match.group(3)}-NOPV"
                span = {
                    "text": f"The CPF case number assigned to this enforcement action is {normalized}.",
                }
                if "page_no" in item:
                    span["page_no"] = item.get("page_no")
                if "line_no" in item:
                    span["line_no"] = item.get("line_no")
                if "paragraph_no" in item:
                    span["paragraph_no"] = item.get("paragraph_no")
                return [span]

        full_text = doc.get("text") or ""
        match = _CPF_RE.search(full_text)
        if match:
            normalized = f"CPF {match.group(1)}-{match.group(2)}-{match.group(3)}-NOPV"
            return [{"text": f"The CPF case number assigned to this enforcement action is {normalized}."}]

        doc_name = doc.get("doc_name") or ""
        match = _DOC_NAME_RE.search(doc_name)
        if match:
            normalized = f"CPF {match.group(1)}-{match.group(2)}-{match.group(3)}-NOPV"
            return [{"text": f"The CPF case number assigned to this enforcement action is {normalized}."}]

        return []
    except Exception:
        return []
