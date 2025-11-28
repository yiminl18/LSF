from sklearn.metrics.pairwise import cosine_similarity
import os, glob
import tiktoken
from openai import AzureOpenAI
import numpy as np
import json
import time
from gpt_4o_azure import gpt_4o_azure

# Azure OpenAI configuration
api_version = "2025-01-01-preview"
azure_endpoint = "https://east-docetl.openai.azure.com/"
with open("/Users/evier/Documents/embedding_key.txt", "r") as f:
    api_key = f.read().strip()
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

# Use your Azure deployment name (not the public model name)
embedding_model = "text-embedding-3-small-3"

def count_tokens(text, model="gpt-4o"):
   encoder = tiktoken.encoding_for_model(model)  # Get the tokenizer for the specific model
   tokens = encoder.encode(text)  # Encode text into tokens
   return len(tokens)

def save_embeddings(filename, embeddings):
    np.save(filename, embeddings)

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def chunking(text, chunk_num = 300):
    """
    Split text into a fixed NUMBER of chunks (by words).
    All chunks except the last have equal size; the last holds the remainder.
    
    Note: `chunk_num` is the NUMBER of chunks to produce.
    """
    words = text.split()
    if not words:
        return []

    # Interpret chunk_num as desired number of chunks
    num_chunks = max(1, min(chunk_num, len(words)))

    base = len(words) // num_chunks
    remainder = len(words) % num_chunks

    chunks = []
    idx = 0
    for i in range(num_chunks):
        take = base if i < num_chunks - 1 else base + remainder
        chunk_words = words[idx:idx + take]
        chunks.append(' '.join(chunk_words))
        idx += take

    return chunks

def test(chunk_num = 300, percentage = 0.03):
    # Paths
    doc_name = '3M_2015_10K.txt'
    doc_path = '/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt/3M_2015_10K.txt'
    question_path = '/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt'

    # Read inputs
    text = open(doc_path, 'r').read()
    with open(question_path, 'r') as f:
        question = f.readline().strip()

    # Retrieval: top K% chunks
    top_chunks = retrieval(doc_path, text, question, percentage, chunk_num)

    # Build prompt
    context = '\n\n---\n\n'.join(top_chunks)
    prompt = (
        'You are a financial analyst. Answer the question using ONLY the context.\n\n'
        f'Context:\n{context}\n\n'
        f'Question: {question}\n'
        'Answer:'
    )

    # Call GPT-4o via Azure and measure latency
    start = time.perf_counter()
    answer = gpt_4o_azure(prompt)
    latency = time.perf_counter() - start

    # Token estimates
    input_token = estimate_tokens(prompt, model="gpt-4o")
    output_token = estimate_tokens(answer, model="gpt-4o")

    # Chunk size (words per chunk, consistent for all except last)
    words = text.split()
    num_chunks = max(1, min(chunk_num, len(words)))
    base_chunk_size = len(words) // num_chunks

    return {
        'Answer': answer,
        'input_token': input_token,
        'output_token': output_token,
        'Latency': latency,
        'Used model': 'gpt-4o',
        'Question': question,
        'Document name': doc_name,
        'K': f'{int(percentage * 100)}%',
        'Chunk size': base_chunk_size,
    }

def _make_key(question, doc_name, p):
    return (question, doc_name, int(p * 100))

def _load_completed_keys(path):
    completed = set()
    if not os.path.exists(path):
        return completed
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get('Question')
                    d = obj.get('Document name')
                    k_str = obj.get('K')  # like '3%'
                    if q and d and k_str and k_str.endswith('%'):
                        completed.add((q, d, int(k_str[:-1])))
                except Exception:
                    continue
    except Exception:
        pass
    return completed

def test_all_pairs(output_path='/Users/evier/PycharmProjects/DocumentSplit/results.jsonl',
                   folder='/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt',
                   questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
                   percentages=(0.01, 0.03, 0.05),
                   chunk_num=300,
                   resume=True,
                   price_per_million_input_tokens=2.5):
    # Read all questions
    with open(questions_path, 'r') as fq:
        questions = [line.strip() for line in fq if line.strip()]

    # List documents
    files = read_files_from_folder(folder)

    # Prepare completion set (and only truncate if not resuming)
    completed = _load_completed_keys(output_path) if resume else set()
    if not resume:
        with open(output_path, 'w') as out:
            pass

    total_q = len(questions)
    total_d = len(files)
    cumulative_cost_usd = 0.0

    for qi, question in enumerate(questions):
        for di, file_info in enumerate(files):
            doc_path = file_info['full_path']
            doc_name = file_info['filename']

            try:
                text = open(doc_path, 'r').read()
            except Exception as e:
                print(f"SKIP read error: Doc={doc_name} | {e}")
                continue

            # Precompute chunk size for reporting
            words = text.split()
            num_chunks = max(1, min(chunk_num, len(words)))
            base_chunk_size = len(words) // num_chunks

            for p in percentages:
                # Only print current pair status
                print(f"Q {qi+1}/{total_q} | Doc {di+1}/{total_d} | K={int(p*100)}% | {doc_name}")

                try:
                    # Skip if already completed
                    if resume and (_make_key(question, doc_name, p) in completed):
                        print(f"SKIP completed: K={int(p*100)}% | {doc_name}")
                        continue
                    # Retrieval
                    top_chunks = retrieval(doc_path, text, question, p, chunk_num)

                    # Prompt
                    context = '\n\n---\n\n'.join(top_chunks)
                    prompt = (
                        'only return the answers, do not add explanation. If answers are not found, return None. \n\n'
                        f'Context:\n{context}\n\n'
                        f'Question: {question}\n'
                        'Answer:'
                    )

                    # Estimate input token cost before API call
                    input_token = estimate_tokens(prompt)
                    cost_usd = input_token * (price_per_million_input_tokens / 1_000_000)
                    cumulative_cost_usd += cost_usd
                    print(f"[COST_EST] Q {qi+1}/{total_q} | Doc {di+1}/{total_d} | K={int(p*100)}% | tokens={input_token} | ${cost_usd:.6f}")
                    print(f"[COST_SUM] ${cumulative_cost_usd:.6f} so far")

                    # Call model with latency
                    start = time.perf_counter()
                    answer = gpt_4o_azure(prompt)
                    latency = time.perf_counter() - start

                    # Token estimates (output only; input already computed above)
                    output_token = estimate_tokens(answer)

                    result = {
                        'Answer': answer,
                        'input_token': input_token,
                        'output_token': output_token,
                        'Latency': latency,
                        'Used model': 'gpt-4o',
                        'Question': question,
                        'Document name': doc_name,
                        'K': f'{int(p * 100)}%',
                        'Chunk size': base_chunk_size,
                    }

                    # Append to JSONL
                    with open(output_path, 'a') as out:
                        out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    if resume:
                        completed.add(_make_key(question, doc_name, p))

                except Exception as e:
                    # Skip this pair on errors
                    print(f"ERROR pair: Q={qi+1}, Doc={doc_name}, K={int(p*100)}% | {e}")

def show_resume_for_test_all_pairs(output_path='/Users/evier/PycharmProjects/DocumentSplit/results.jsonl',
                                   folder='/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt',
                                   questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
                                   percentages=(0.01, 0.03, 0.05),
                                   chunk_num=300):
    with open(questions_path, 'r') as fq:
        questions = [line.strip() for line in fq if line.strip()]
    files = read_files_from_folder(folder)
    total = len(questions) * len(files) * len(percentages)
    completed = _load_completed_keys(output_path)

    # Find next missing
    next_missing = None
    for q in questions:
        for f in files:
            for p in percentages:
                key = (q, f['filename'], int(p * 100))
                if key not in completed:
                    next_missing = {
                        'Question': q,
                        'Document name': f['filename'],
                        'K': f'{int(p*100)}%'
                    }
                    break
            if next_missing:
                break
        if next_missing:
            break

    print(f"[RESUME] completed={len(completed)} / total={total}")
    if next_missing:
        print(f"[RESUME] next: {next_missing['K']} | {next_missing['Document name']}")
    else:
        print("[RESUME] all done")
        #break

def estimate_cost_all_pairs(output_path='/Users/evier/PycharmProjects/DocumentSplit/costs.jsonl',
                            folder='/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt',
                            questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
                            percentages=(0.01, 0.03, 0.05),
                            chunk_num=300,
                            price_per_million_input_tokens=2.5):
    """
    Estimate GPT-4o prompt cost for all (question, document) pairs without making completion calls.
    Cost is computed as input_tokens * (price_per_million_input_tokens / 1_000_000).
    """
    # Read all questions
    with open(questions_path, 'r') as fq:
        questions = [line.strip() for line in fq if line.strip()]

    # List documents
    files = read_files_from_folder(folder)

    # Prepare output file (truncate)
    with open(output_path, 'w') as out:
        pass

    total_q = len(questions)
    total_d = len(files)

    for qi, question in enumerate(questions):
        for di, file_info in enumerate(files):
            doc_path = file_info['full_path']
            doc_name = file_info['filename']

            try:
                text = open(doc_path, 'r').read()
            except Exception as e:
                print(f"SKIP read error: Doc={doc_name} | {e}")
                continue

            for p in percentages:
                # Only print current pair status
                print(f"[COST] Q {qi+1}/{total_q} | Doc {di+1}/{total_d} | K={int(p*100)}% | {doc_name}")

                try:
                    # Get top-k% chunks using existing retrieval logic
                    top_chunks = retrieval(doc_path, text, question, p, chunk_num)

                    # Build prompt
                    context = '\n\n---\n\n'.join(top_chunks)
                    prompt = (
                        'You are a financial analyst. Answer the question using ONLY the context.\n\n'
                        f'Context:\n{context}\n\n'
                        f'Question: {question}\n'
                        'Answer:'
                    )

                    input_tokens = estimate_tokens(prompt)
                    cost_usd = input_tokens * (price_per_million_input_tokens / 1_000_000)

                    record = {
                        'Question': question,
                        'Document name': doc_name,
                        'K': f'{int(p * 100)}%',
                        'input_token': input_tokens,
                        'estimated_cost_usd': cost_usd,
                        'Used model': 'gpt-4o'
                    }

                    with open(output_path, 'a') as out:
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"[COST][ERROR] Q={qi+1}, Doc={doc_name}, K={int(p*100)}% | {e}")

def estimate_cost_docs_only(output_path='/Users/evier/PycharmProjects/DocumentSplit/costs_docs_only.jsonl',
                            folder='/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt',
                            questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
                            percentages=(0.01, 0.03, 0.05),
                            chunk_num=300,
                            price_per_million_input_tokens=2.5):
    """
    Estimate GPT-4o input token cost using ONLY document chunks (no retrieval, no API).
    For each document and K in percentages, we sum the token count of the first K% chunks,
    then multiply by the number of questions.
    """
    # Read all questions (count only)
    with open(questions_path, 'r') as fq:
        questions = [line.strip() for line in fq if line.strip()]
    num_questions = len(questions)

    files = read_files_from_folder(folder)

    # Truncate output file
    with open(output_path, 'w') as out:
        pass

    total_d = len(files)
    for di, file_info in enumerate(files):
        doc_path = file_info['full_path']
        doc_name = file_info['filename']

        try:
            text = open(doc_path, 'r').read()
        except Exception as e:
            print(f"[COST_DOCS][SKIP] Doc={doc_name} | {e}")
            continue

        chunks = chunking(text, chunk_num)
        if not chunks:
            continue

        # Precompute token counts per chunk
        chunk_tokens = [estimate_tokens(c) for c in chunks]
        num_chunks = len(chunks)

        for p in percentages:
            k = max(1, int(num_chunks * p))
            selected_tokens = sum(chunk_tokens[:k])
            total_tokens_all_pairs = selected_tokens * num_questions
            cost_usd = total_tokens_all_pairs * (price_per_million_input_tokens / 1_000_000)

            print(f"[COST_DOCS] Doc {di+1}/{total_d} | K={int(p*100)}% | {doc_name}")

            record = {
                'Document name': doc_name,
                'K': f'{int(p * 100)}%',
                'num_questions': num_questions,
                'input_tokens_per_pair': selected_tokens,
                'total_input_tokens_all_pairs': total_tokens_all_pairs,
                'estimated_cost_usd': cost_usd,
                'Used model': 'gpt-4o'
            }

            with open(output_path, 'a') as out:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')

def sum_costs(jsonl_path='/Users/evier/PycharmProjects/DocumentSplit/costs_docs_only.jsonl'):
    total = 0.0
    count = 0
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cost = obj.get('estimated_cost_usd', 0) or 0
                    total += float(cost)
                    count += 1
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"[SUM][MISS] {jsonl_path} not found")
        return 0.0
    print(f"[SUM] Total cost (USD): {total:.6f} over {count} records")
    return total

def load_embeddings(filename):
    return np.load(filename, allow_pickle=True).item()

def build_embeddings(file_path, text, chunk_num = 300):
    embeddings = {}
    i = 0
    chunks = chunking(text, chunk_num)
    for chunk in chunks:
    #    print(i, len(chunks))
        if chunk not in embeddings:
            embeddings[chunk] = get_embedding(chunk)
        i += 1

    save_embeddings(file_path, embeddings)

def retrieval(file_path, text, question, percentage, chunk_num = 300): 
    try:
        embeddings = load_embeddings(file_path + ".npy")
        exist = True
        print('Embeddings found!')
    except (FileNotFoundError, OSError, ValueError):
        embeddings = {}
        exist = False
        print('Embeddings not found!')
    
    question_embedding = get_embedding(question) 
        
    # Compute embeddings and similarity scores for each sentence
    similarity_scores = {}
    i = 0
    chunks = chunking(text, chunk_num)
    for chunk in chunks:
     #   print(i, len(chunks))
        if chunk not in embeddings:
            embeddings[chunk] = get_embedding(chunk)
        chunk_embedding = embeddings[chunk]
        similarity = cosine_sim(question_embedding, chunk_embedding)
        similarity_scores[chunk] = similarity
        i += 1
    # Save updated embeddings
    if not exist:
        print('Saving embeddings...')
        save_embeddings(file_path + ".npy", embeddings)

    indexed_sentences = list(enumerate(chunks))
    sorted_indexed_sentences = sorted(indexed_sentences, key=lambda x: similarity_scores[x[1]], reverse=True)


   # Extract the sorted sentences and their original indices
    sorted_chunks = [chunk for index, chunk in sorted_indexed_sentences]
    sorted_indices = [index for index, chunk in sorted_indexed_sentences]

   # Return the top percentage of chunks
    k = max(1, int(len(sorted_chunks) * percentage))
    return sorted_chunks[:k]



def read_files_from_folder(folder_path, suffix=".txt"):
    files = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(file_path) and file_path.endswith(suffix):
            filename = os.path.basename(file_path)
            files.append({
                'filename': filename,
                'full_path': file_path
            })
    return files

def estimate_tokens(text, model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(model)  # Get the tokenizer for the specific model
    tokens = encoder.encode(text)  # Encode text into tokens
    return len(tokens)


if __name__ == "__main__":
    folder = '/Users/evier/PycharmProjects/DocumentSplit/financebench_raw_txt'
    # Run batch test and only print current pair status per iteration
    
    test_all_pairs(
        output_path='/Users/evier/PycharmProjects/DocumentSplit/results.jsonl',
        folder=folder,
        questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
        percentages=(0.01, 0.03, 0.05),
        chunk_num=300,
    )
    
    '''
    estimate_cost_all_pairs(
        output_path='/Users/evier/PycharmProjects/DocumentSplit/costs.jsonl',
        folder=folder,
        questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
        percentages=(0.01, 0.03, 0.05),
        chunk_num=300,
    )
    '''
    '''
    estimate_cost_docs_only(
        output_path='/Users/evier/PycharmProjects/DocumentSplit/costs_docs_only.jsonl',
        folder=folder,
        questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
        percentages=(0.01, 0.03, 0.05),
        chunk_num=300,
    )

    sum_costs('/Users/evier/PycharmProjects/DocumentSplit/costs_docs_only.jsonl')

    # Show resume status for test_all_pairs so you know where to continue
    show_resume_for_test_all_pairs(
        output_path='/Users/evier/PycharmProjects/DocumentSplit/results.jsonl',
        folder=folder,
        questions_path='/Users/evier/PycharmProjects/DocumentSplit/10k_factual_questions.txt',
        percentages=(0.01, 0.03, 0.05),
        chunk_num=300,
    )
    '''
    