#!/usr/bin/env python3
"""Sanity check: locate the EXIT Gemma-2B PEFT adapter and test one classification.

Run from repo root:
    PYTHONPATH=src python3 scripts/check_exit_checkpoint.py

If torch is not installed, this script prints the checkpoint path and exits
without loading the model (safe for CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.baselines.exit.classifier import _find_gemma_checkpoint, _gemma_checkpoint_available

checkpoint_path = _find_gemma_checkpoint()
if checkpoint_path is None:
    print("ERROR: No EXIT Gemma checkpoint found.")
    print("Options:")
    print("  1. Place adapter_config.json + adapter_model.safetensors under")
    print("     src/agent/baselines/exit/upstream/EXIT/checkpoints/<name>/")
    print("  2. Set EXIT_CHECKPOINT_DIR=/path/to/adapter/dir")
    print("  3. Download via: huggingface-cli download doubleyyh/exit-gemma-2b")
    sys.exit(1)

print(f"Found checkpoint at: {checkpoint_path}")
print(f"Files: {[f.name for f in checkpoint_path.iterdir()]}")

# Check whether torch is available before trying to load
try:
    import torch  # noqa: F401
except ImportError:
    print("torch not installed — skipping model load. Checkpoint path is valid.")
    sys.exit(0)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import PeftModel  # type: ignore[import]
except ImportError as e:
    print(f"transformers/peft not installed ({e}) — skipping model load.")
    sys.exit(0)

print("Loading base model google/gemma-2b-it (this may take a minute) ...")
base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, str(checkpoint_path))
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
print("Model loaded successfully.")

# One-shot classification
query = "What is the company name?"
sentence = "Amazon was founded by Jeff Bezos."
prompt = (
    f"<start_of_turn>user\n"
    f"Query:\n{query}\n"
    f"Sentence:\n{sentence}\n"
    f'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
    f"<end_of_turn>\n<start_of_turn>model\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False, temperature=1.0)
decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Query: {query!r}")
print(f"Sentence: {sentence!r}")
print(f"Classification: {decoded.strip()!r}")
print("Sanity check PASSED.")
