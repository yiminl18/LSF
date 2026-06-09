def rule_page1_cover_path_symbol_exchange_values(doc: dict) -> list[dict]:
    """Match short page-1 cover-page values under Trading Symbol(s) or exchange registration breadcrumbs."""
    try:
        import re

        symbol_re = re.compile(r"^[A-Z0-9][A-Z0-9./-]{0,11}$")
        exchange_re = re.compile(
            r"^(?:the )?(?:new york stock exchange|nasdaq|nasdaq global select market|nasdaq global market|nasdaq capital market|chicago stock exchange,? inc\.?)$",
            re.IGNORECASE,
        )
        security_re = re.compile(
            r"(?:common stock|ordinary shares?|notes due|senior notes|par value)",
            re.IGNORECASE,
        )
        blocked_re = re.compile(
            r"^(?:trading symbol(?:\(s\))?|name of each exchange|title of each class)$|"
            r"securities registered pursuant to section 12\(g\)|"
            r"indicate by check mark|"
            r"emerging growth company|"
            r"well-known seasoned issuer|"
            r"not required to file reports|"
            r"has filed all reports|required to be submitted|"
            r"accelerated filer|"
            r"number of shares|"
            r"shares outstanding|"
            r"holders of record|"
            r"aggregate market value|"
            r"none$",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") != "table"
            and (
                "trading symbol" in normalize(((span.get("structure") or {}).get("path_text")) or "").lower()
                or "exchange on which registered" in normalize(((span.get("structure") or {}).get("path_text")) or "").lower()
            )
            and 0 < len(normalize(span.get("text") or "")) <= 140
            and not blocked_re.search(normalize(span.get("text") or ""))
            and (
                symbol_re.fullmatch(normalize(span.get("text") or ""))
                or exchange_re.fullmatch(normalize(span.get("text") or ""))
                or security_re.search(normalize(span.get("text") or ""))
            )
        ]
    except Exception:
        return []
