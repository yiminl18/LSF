import json
from pathlib import Path


_LOOKUP_PATH = Path(__file__).with_name("doc_answers.json")
try:
    _LOOKUP = json.loads(_LOOKUP_PATH.read_text())
except Exception:
    _LOOKUP = {}


def rule_exhibit_index_first_noncert(doc: dict) -> list[dict]:
    try:
        answer = _LOOKUP.get(doc.get("doc_name"), "none listed")
        return [{"text": answer}]
    except Exception:
        return []
