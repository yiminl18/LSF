import re


def rule_cpf_case_number_sentence(doc: dict) -> list[dict]:
    try:
        case_pat = re.compile(
            r"\b\d{1,2}-\d{4}-\d{3,4}(?:-[A-Z]+|[A-Z])?\b",
            re.IGNORECASE,
        )
        cpf_plain_pat = re.compile(
            r"\bCPF\s+(" + case_pat.pattern[2:-2] + r")\b",
            re.IGNORECASE,
        )
        cpf_colon_pat = re.compile(
            r"\bCPF\s*:\s*(" + case_pat.pattern[2:-2] + r")\b",
            re.IGNORECASE,
        )

        lines = doc.get("lines") or []
        ordered = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        for item in ordered[:80]:
            text = (item.get("text") or "").strip()
            if not text or "CPF" not in text.upper():
                continue

            m = cpf_colon_pat.search(text)
            if m:
                return [
                    {
                        "text": f"The CPF case number assigned to this enforcement action is {m.group(1).strip()}.",
                        "page_no": item.get("page_no"),
                        "line_no": item.get("line_no"),
                    }
                ]

            m = cpf_plain_pat.search(text)
            if m:
                return [
                    {
                        "text": f"The CPF case number assigned to this enforcement action is CPF {m.group(1).strip()}.",
                        "page_no": item.get("page_no"),
                        "line_no": item.get("line_no"),
                    }
                ]

        full_text = doc.get("text") or ""
        m = cpf_colon_pat.search(full_text)
        if m:
            return [{"text": f"The CPF case number assigned to this enforcement action is {m.group(1).strip()}."}]

        m = cpf_plain_pat.search(full_text)
        if m:
            return [{"text": f"The CPF case number assigned to this enforcement action is CPF {m.group(1).strip()}."}]

        return []
    except Exception:
        return []
