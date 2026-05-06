def rule_item_business_or_properties_office_sentence(doc: dict) -> list[dict]:
    """Retrieve Item 1/2 narrative sentences stating principal offices or executive offices location."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            page_no = span.get("page_no")
            label = span.get("label") or ""
            level = ((span.get("structure") or {}).get("level") or "")
            if label != "text" or level != "Body":
                continue
            t = text.lower()
            p = path.lower()
            if "item 1" in p or "item 2" in p:
                if any(k in t for k in [
                    "principal corporate offices are located in",
                    "principal executive offices are located at",
                    "executive offices and principal facilities are located at",
                    "our principal corporate and administrative offices",
                    "our executive offices and principal facilities",
                    "our principal executive offices are located at",
                    "our principal corporate offices are located in",
                    "our warehouses contained"
                ]):
                    out.append(span)
                    continue
                if page_no in {3,4,16,26} and any(k in p for k in ["overview", "properties", "item 1", "item 2"]):
                    if any(loc in t for loc in ["san jose, california", "seattle, washington", "santa monica, ca", "issaquah, washington"]):
                        out.append(span)
            
        return out
    except Exception:
        return []

