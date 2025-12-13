import argparse
from typing import Tuple

import tiktoken
import json

from core.ask import ask
from core.gpt_4o_azure import gpt_4o_azure


def normalize_exact(s: str) -> str:
    """Lowercase/strip and remove simple punctuation for exact compare."""
    if not s:
        return ""
    cleaned = s.strip().lower()
    cleaned = cleaned.replace(".", "").replace(",", "")
    return cleaned


def to_text(value) -> str:
    """Convert non-string ground truth/response to string for comparison."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def equal_llm(res1: str, res2: str, question: str) -> Tuple[bool, int]:
    """
    复制自 evaluate_baseline.py 的等价判定逻辑，但本文件内独立实现。
    返回 (是否等价, 估算消耗的输入token数)
    """
    instruction = (
        "I have two answers to the given question. If these two answers are "
        "equivalent in meaning, return True; otherwise, return False. Ignore minor wording differences. "
        "Do not provide any explanation. "
        + "Answer 1: " + str(res1) + " Answer 2: " + str(res2) + " Question: " + str(question)
    )
    tokens = estimate_tokens(instruction)
    try:
        resp = gpt_4o_azure(instruction, key_path="/Users/evier/Documents/gpt-4o.txt", max_tokens=8, temperature=0)
        resp_str = resp if isinstance(resp, str) else str(resp)
        if "true" in resp_str.lower():
            return True, tokens
    except Exception:
        pass
    return False, tokens


def judge_header(text: str, question: str, ground_truth: str, key_path: str) -> Tuple[bool, str]:
    """
    先调用 ask 获取回答，再执行归一化判定：
      1) 若归一化后为空或“none”则直接不匹配
      2) 归一化精确相等则匹配
      3) 否则调用 equal_llm 语义判定
    返回 (是否匹配, ask原始回答)
    """
    predicted_raw = ask(text, question, key_path=key_path)
    predicted = to_text(predicted_raw)
    gt_text = to_text(ground_truth) if ground_truth is not None else ""
    predicted_norm = normalize_exact(predicted)
    gt_norm = normalize_exact(gt_text)

    # 1) 先做归一化文本比较
    if predicted_norm and gt_norm and predicted_norm == gt_norm:
        return True, predicted

    # 2) 若回答为空或 none，判不匹配
    if not predicted_norm or predicted_norm == "none":
        return False, predicted

    # 3) 若 GT 为 None 或 none，不直接放行，继续走等价判定
    is_eq, _ = equal_llm(predicted or "", ground_truth, question)
    return is_eq, predicted


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge a single header text against a question/answer.")
    parser.add_argument("--text", required=True, help="Header text (or combined text_span).")
    parser.add_argument("--question", required=True, help="Question string.")
    parser.add_argument("--answer", required=True, help="Ground-truth answer string.")
    parser.add_argument(
        "--key-path",
        default="/Users/evier/Documents/gpt-4o.txt",
        help="Path to GPT key file for ask().",
    )
    args = parser.parse_args()

    matched, resp = judge_header(args.text, args.question, args.answer, args.key_path)
    print(f"matched={matched}")
    print(f"llm_response={resp}")


if __name__ == "__main__":
    main()

