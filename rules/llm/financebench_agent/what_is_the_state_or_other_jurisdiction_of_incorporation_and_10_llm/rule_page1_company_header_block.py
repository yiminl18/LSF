def rule_page1_company_header_block(doc: dict) -> list[dict]:
    """Retrieve the compact page-1 company header block containing incorporation state and EIN."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_roots = set()
        for span in texts:
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = text.lower()
            p = path.lower()
            if (
                "exact name of registrant" in t
                or "state or other jurisdiction of incorporation" in t
                or "state of incorporation" in t
                or "employer identification no" in t
                or "exact name of registrant" in p
                or "state or other jurisdiction of incorporation" in p
                or "state of incorporation" in p
                or "employer identification no" in p
            ):
                root = path.split("|")[0].strip() if path else ""
                if root:
                    company_roots.add(root)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            root = path.split("|")[0].strip() if path else ""
            if root and root in company_roots:
                out.append(span)
        return out
    except Exception:
        return []

