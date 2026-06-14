def rule_signature_block_after_digital_or_embedded_sincerely(doc: dict) -> list[dict]:
    """Match digital-signature or embedded-Sincerely blocks on later pages."""
    try:
        import re

        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            path_text = ((span.get("structure") or {}).get("path_text") or "").strip()
            combined = f"{text} {path_text}".lower()
            if span.get("page_no", 0) < 2:
                continue
            if "digitally signed by" not in combined and not combined.startswith("sincerely,"):
                continue

            page_no = span.get("page_no")
            for next_span in texts[i:i + 6]:
                next_text = (next_span.get("text") or "").strip()
                if next_span.get("page_no") != page_no:
                    break
                if not next_text:
                    continue
                if (
                    re.search(r"\b(?:Acting\s+)?Director\b", next_text, re.I)
                    or re.search(r"Pipeline and Hazardous Materials Safety Administration", next_text, re.I)
                    or re.match(r"^[A-Z][A-Za-z.\-'\s]{3,80}$", next_text)
                ):
                    out.append(next_span)
        return out
    except Exception:
        return []
