def rule_page23_body_location_state_mentions(doc: dict) -> list[dict]:
    """Match page-2/3 inspection body spans that mention a specific facility location with a state token."""
    try:
        import re

        state_re = re.compile(
            r"\b(?:Alabama|Alaska|Arizona|California|Colorado|Florida|Hawaii|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Mississippi|Nebraska|New Mexico|North Dakota|Ohio|Oklahoma|Pennsylvania|Puerto Rico|Tennessee|Texas|Virginia|Wyoming|AK|AL|AZ|CA|CO|FL|HI|IA|IL|IN|KS|KY|LA|MS|ND|NE|NM|OH|OK|PA|PR|TN|TX|VA|WY)\b",
            re.I,
        )
        location_re = re.compile(
            r"\b(?:county|terminal|pump station|compressor station|meter station|site|facility)\b",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = " ".join((span.get("text") or "").split())
            lowered = text.lower()
            if span.get("label") != "text" or not (2 <= span.get("page_no", 0) <= 3):
                continue
            if "during the inspection" not in lowered and "phmsa requested" not in lowered:
                continue
            if not location_re.search(text) or not state_re.search(text):
                continue
            out.append(span)
        return out
    except Exception:
        return []
