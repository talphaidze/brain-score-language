"""
Sweep Apertus-8B layers on LeBel2023 benchmark.

Usage:
    python sweep_apertus_layers_lebel.py
"""
import sys
sys.path.insert(0, '/mnt/alphaidz/litcoder_core')

import gc
import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

# --- Patch data loader ---
from test_lebel_local import load_lebel_from_pickle
import brainscore_language
from brainscore_language.benchmarks.lebel2023 import benchmark as lebel_benchmark_module

PKL_PATH = '/mnt/alphaidz/data/assembly_lebel_uts03.pkl'
_original_load_dataset = brainscore_language.load_dataset
MAX_STORIES = 25

def _patched_load_dataset(identifier):
    if identifier == 'LeBel.fROI':
        return load_lebel_from_pickle(PKL_PATH, max_stories=MAX_STORIES)
    return _original_load_dataset(identifier)

brainscore_language.load_dataset = _patched_load_dataset
lebel_benchmark_module.load_dataset = _patched_load_dataset

from brainscore_language.benchmarks.lebel2023.benchmark import LeBelRidge

# --- Config ---
MODEL_ID = 'swiss-ai/Apertus-8B-2509'
NUM_LAYERS = 32
TEST_STORY = 'wheretheressmoke'
SAVE_PATH = 'sweep_results_lebel_local.json'

# --- Load previous results if resuming ---
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'r') as f:
        results = {int(k): v for k, v in json.load(f).items()}
    print(f"Loaded {len(results)} previous results from {SAVE_PATH}")
else:
    results = {}

# --- Load model once ---
print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Sweep a subset first, then narrow down
layers_to_sweep = [6, 7, 9, 10]#list(range(12, NUM_LAYERS, 4))  # [0, 4, 8, 12, 16, 20, 24, 28]
# layers_to_sweep = list(range(NUM_LAYERS))  # all 32

for layer_idx in layers_to_sweep:
    if layer_idx in results:
        print(f"\n=== Layer {layer_idx} (already done, skipping) ===")
        continue

    layer_name = f'model.layers.{layer_idx}'
    print(f"\n=== Layer {layer_idx} ({layer_name}) ===")

    subject = HuggingfaceSubject(
        model_id=MODEL_ID,
        model=model,
        tokenizer=tokenizer,
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: layer_name
        },
    )

    benchmark = LeBelRidge(test_story=TEST_STORY)
    score = benchmark(subject)
    raw = float(score.attrs['raw'].values)
    normalized = float(score.values)
    results[layer_idx] = {'raw': raw, 'normalized': normalized}
    print(f"  Raw: {raw:.4f} | Ceiling-normalized: {normalized:.4f}")

    # Save after each layer
    with open(SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {SAVE_PATH}")

    # Free GPU memory
    del subject, benchmark, score
    gc.collect()
    torch.cuda.empty_cache()

# --- Summary ---
print("\n" + "=" * 60)
print(f"{'Layer':<25} {'Raw':>10} {'Normalized':>12}")
print("-" * 60)

best_layer = None
best_raw = -float('inf')

for layer_idx in sorted(results.keys()):
    r = results[layer_idx]
    marker = ""
    if r['raw'] > best_raw:
        best_raw = r['raw']
        best_layer = layer_idx
        marker = " <-- best"
    print(f"model.layers.{layer_idx:<13} {r['raw']:>10.4f} {r['normalized']:>12.4f}{marker}")

print("=" * 60)
print(f"\nBest layer: model.layers.{best_layer} (raw: {best_raw:.4f})")
print(f"Results saved to {SAVE_PATH}")
