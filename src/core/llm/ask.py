"""
基于上下文的问答功能

使用 LLM 根据给定的上下文文本回答问题，支持多 provider。

主要功能：
- ask(): 根据上下文回答问题

依赖：
- core.llm.model: 统一 LLM 调度
"""

from core.llm.model import llm_call


def ask(
    text: str,
    question: str,
    llm_provider: str = "azure",
    *,
    model: str,
) -> str:
    """
    基于上下文文本回答问题。

    参数:
        text: 用于回答问题的上下文文本
        question: 要回答的问题
        llm_provider: LLM 提供商（"azure" 或 "openrouter"）
        model: LLM 模型名称，必须显式指定

    返回:
        答案字符串
    """
    # 修改 prompt 以避免触发 Azure 内容过滤器（移除可能被误判为 jailbreak 的指令）
    prompt = (
        "Based on the context provided below, answer the question. "
        "If the answer is not in the context, respond with 'None'.\n\n"
        f"Context:\n{text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return llm_call(prompt, llm_provider=llm_provider, model=model)


if __name__ == "__main__":
    context = "Microsoft Corporation is a technology company."
    question = "What is Microsoft?"
    answer = ask(context, question, model="gpt-5.4-mini")
    print(f"Answer: {answer}")
