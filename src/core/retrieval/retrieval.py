"""
Retrieval Module

Handles finding and ranking provenance nodes using embeddings and LLM verification.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.embed.embeddings import (
    load_document_embeddings,
    get_query_embedding,
    cosine_sim,
    get_combined_text,
    get_model_name_for_provider,
)
from core.retrieval.judge_header import judge_header, ContentFilterError


def normalize_header_for_provenance(
    header: Dict[str, Any], header_idx: int
) -> Optional[Dict[str, Any]]:
    """Normalize header per generate_labels rules; returns filter reason when ineligible."""
    from core.config import MAX_SPAN_WORDS

    if header.get("label") != "section_header":
        return {"eligible": False, "reason": "NotSectionHeader"}
    if "table_data" in header:
        return {"eligible": False, "reason": "TableNode"}

    text_span = header.get("text_span", "")
    if len(text_span.split()) >= MAX_SPAN_WORDS:
        return {"eligible": False, "reason": "SpanTooLong"}

    combined_text = get_combined_text(header)
    if not combined_text:
        return {"eligible": False, "reason": "EmptyCombinedText"}

    path_text = (header.get("structure") or {}).get("path_text", "") or header.get(
        "path_text", ""
    )
    return {
        "eligible": True,
        "header_idx": header_idx,
        "text": header.get("text", ""),
        "text_span": text_span,
        "path_text": path_text,
        "combined_text": combined_text,
    }


def find_provenance_node(
    pdf_path: str,
    merged_json_path: str,
    question: str,
    answer: str,
    cache_dir: Optional[str] = None,
    provider: str = "openai",
    top_k_check: int = 50,
    judge_mode: str = "answer_compare",
    llm_provider: str = "azure",
    llm_model: Optional[str] = None,
    match_limit: int = 3,
    header_page: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Find headers that answer the question using embedding retrieval and LLM verification.

    1. Calculate cosine similarity between query and all headers.
    2. Sort headers by similarity.
    3. Iterate through sorted headers:
       - Check if header answers question using LLM (judge_header).
       - If match found, add to results.
       - Stop after finding match_limit matches or exceeding top_k_check.

    Returns:
        (matches, total_checked_count)
    """
    if not llm_model:
        raise ValueError("llm_model must be specified explicitly")

    print(f"Processing PDF (Unified Provenance): {pdf_path}")

    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    headers = merged_data.get("texts", [])
    if not headers:
        print("No headers found")
        return [], 0

    # Load embeddings
    document_embeddings, _ = load_document_embeddings(
        merged_json_path, cache_dir, provider=provider
    )

    # Get query embedding
    model = get_model_name_for_provider(provider)
    question_embedding = get_query_embedding(question, model=model, provider=provider)

    # Calculate similarities
    header_similarities = []
    for idx, header in enumerate(headers):
        # Optional: only search headers on the specified page
        if header_page is not None:
            current_page = header.get("header_page")
            if current_page is None:
                current_page = header.get("page_no")
            if current_page != header_page:
                continue

        normalized = normalize_header_for_provenance(header, idx)
        if not normalized or not normalized["eligible"]:
            continue
        combined_text = normalized["combined_text"]

        header_embedding = document_embeddings.get(combined_text)
        if header_embedding is None:
            header_embedding = [0.0] * len(question_embedding)

        similarity = cosine_sim(header_embedding, question_embedding)
        header_similarities.append((similarity, header, combined_text, idx))

    header_similarities.sort(key=lambda x: x[0], reverse=True)

    matched_nodes = []
    top1_sim = header_similarities[0][0] if header_similarities else 0.0
    # last_checked_rank: highest rank actually checked (1-based); 0 if header_similarities is empty
    last_checked_rank = 0
    # Mark whether a match was found within top-k
    found_in_top_k = False

    def get_concurrent_size(rank_idx: int) -> int:
        """Return concurrent request count based on rank (escalating strategy)."""
        if rank_idx < 10:
            return 1  # Top 10: sequential
        elif rank_idx < 30:
            return 2  # rank 11-30: concurrency 2
        elif rank_idx < 60:
            return 4  # rank 31-60: concurrency 4
        else:
            return 8  # rank 61+: concurrency 8

    rank_idx = 0
    while rank_idx < len(header_similarities):
        # Stop condition: Found enough matches
        if len(matched_nodes) >= match_limit:
            break

        # Stop condition: Past Top K and already found a match
        # If past top_k_check with existing matches, stop checking (avoid unnecessary LLM calls)
        if rank_idx >= top_k_check and len(matched_nodes) > 0:
            break

        # Determine concurrency level
        concurrent_size = get_concurrent_size(rank_idx)
        batch_end = min(rank_idx + concurrent_size, len(header_similarities))

        # Collect items for concurrent processing
        batch_items = []
        batch_indices = []
        for i in range(rank_idx, batch_end):
            similarity, header, combined_text, idx = header_similarities[i]
            path_text = (header.get("structure") or {}).get(
                "path_text", ""
            ) or header.get("path_text", "")
            batch_items.append((i, similarity, header, idx, combined_text, path_text))
            batch_indices.append(i)

        # Concurrent processing
        try:
            if concurrent_size == 1:
                # Sequential processing (uses original logic, supports caching)
                i, similarity, header, idx, combined_text, path_text = batch_items[0]
                is_match, predicted_answer = judge_header(
                    combined_text,
                    question,
                    answer,
                    mode=judge_mode,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    path_text=path_text,
                )
                results = [(i, similarity, header, idx, is_match, predicted_answer)]
            else:
                # Concurrent processing: use ThreadPoolExecutor to send multiple independent API requests
                results = []
                with ThreadPoolExecutor(max_workers=concurrent_size) as executor:
                    # Submit all tasks
                    future_to_item = {}
                    for (
                        i,
                        similarity,
                        header,
                        idx,
                        combined_text,
                        path_text,
                    ) in batch_items:
                        future = executor.submit(
                            judge_header,
                            combined_text,
                            question,
                            answer,
                            mode=judge_mode,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            path_text=path_text,
                        )
                        future_to_item[future] = (i, similarity, header, idx)

                    # Collect results (in submission order)
                    for future in as_completed(future_to_item):
                        i, similarity, header, idx = future_to_item[future]
                        try:
                            is_match, predicted_answer = future.result()
                            results.append(
                                (i, similarity, header, idx, is_match, predicted_answer)
                            )
                        except ContentFilterError as e:
                            # Content filter error: re-raise immediately
                            raise ContentFilterError(
                                f"Content filter triggered while processing document {pdf_path}, "
                                f"question: {question[:100]}..., header at rank {i + 1}"
                            ) from e
                        except Exception as e:
                            # Any error: re-raise immediately, stop processing
                            raise RuntimeError(
                                f"Error judging header at rank {i + 1} for document {pdf_path}, "
                                f"question: {question[:100]}...: {e}"
                            ) from e

                # Sort results by rank order (concurrent execution order may differ)
                results.sort(key=lambda x: x[0])
        except ContentFilterError:
            # Re-raise ContentFilterError
            raise
        except Exception as e:
            # Any error: re-raise immediately, stop processing
            raise RuntimeError(
                f"Error judging headers at rank {rank_idx + 1}-{batch_end} for document {pdf_path}, "
                f"question: {question[:100]}...: {e}"
            ) from e

        # Process results
        for i, similarity, header, idx, is_match, predicted_answer in results:
            last_checked_rank = i + 1  # Update to 1-based

            if is_match:
                gap = top1_sim - similarity
                margin = 0.0
                if i < len(header_similarities) - 1:
                    next_sim = header_similarities[i + 1][0]
                    margin = similarity - next_sim

                matched_nodes.append(
                    {
                        "header": header,
                        "header_idx": idx,
                        "rank_of_accepted": i + 1,
                        "similarity": similarity,
                        "gap_from_top1": gap,
                        "margin_to_next": margin,
                    }
                )

                # Mark whether a match was found within top-k
                if i < top_k_check:
                    found_in_top_k = True

                # If first match found after top_k_check (and none found in top-k), stop immediately
                if i >= top_k_check and not found_in_top_k:
                    return matched_nodes, last_checked_rank

        # Move to next batch
        rank_idx = batch_end

    return matched_nodes, last_checked_rank
