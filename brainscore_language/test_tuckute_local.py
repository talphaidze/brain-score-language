"""
Test Tuckute2024 benchmark locally using the CSV data file,
bypassing S3.

Usage:
    python test_tuckute_local.py
"""

# --- Config ---
CSV_PATH = '/mnt/alphaidz/data/brain-lang-data_participant_20230728.csv'
MODEL = 'apertus-8b'

# --- Patch load_dataset to bypass S3 ---
import brainscore_language
from brainscore_language.data.tuckute2024.data_packaging import load_tuckute2024_5subj

_original_load_dataset = brainscore_language.load_dataset

def _patched_load_dataset(identifier):
    if identifier == 'Tuckute2024.language':
        return load_tuckute2024_5subj(source=CSV_PATH)
    return _original_load_dataset(identifier)

brainscore_language.load_dataset = _patched_load_dataset

from brainscore_language.benchmarks.tuckute2024 import benchmark as tuckute_benchmark_module
tuckute_benchmark_module.load_dataset = _patched_load_dataset

# --- Load benchmark ---
from brainscore_language.benchmarks.tuckute2024.benchmark import Tuckute2024_ridge
benchmark = Tuckute2024_ridge()
print(f"Benchmark loaded: {benchmark.identifier}")
print(f"Data shape: {benchmark.data.shape}")
print(f"Stimuli sample: {benchmark.data['stimulus'].values[:3]}")

# --- Run benchmark ---
from brainscore_language import load_model
model = load_model(MODEL)

score = benchmark(model)
print(f"\n=== RESULTS ===")
if isinstance(score, dict):
    for layer, s in score.items():
        print(f"  {layer}: {s.values:.4f}")
    score = max(iter(score.values()))
print(f"Score: {score}")
print(f"Raw attributes: {score.attrs}")
