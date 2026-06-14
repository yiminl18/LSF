def rule_signature_block_after_standalone_sincerely(doc: dict) -> list[dict]:
    """Match the later-page signature block that follows a standalone Sincerely line."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(
            r"^(?:Enclosures?:|cc:|cC:|Proposed Compliance Order$|Response Options|PROPOSED COMPLIANCE ORDER)",
            re.I,
        )
        sincerely_re = re.compile(r"^sincerely,?$", re.I)

        out = []
        for i, span in enumerate(texts):
            text = (span.get("text") or "").strip()
            if span.get("page_no", 0) < 2 or not sincerely_re.match(text):
                continue

            page_no = span.get("page_no")
            block = []
            for next_span in texts[i + 1:i + 7]:
                next_text = (next_span.get("text") or "").strip()
                if next_span.get("page_no") != page_no:
                    break
                if stop_re.match(next_text):
                    break
                if next_text:
                    block.append(next_span)
            if block:
                out.extend(block)
        return out
    except Exception:
        return []
