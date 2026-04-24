"""
Test Narratives benchmark locally using the pickle assembly from data,
bypassing S3. Uses KFold CV (no shuffle) on TRs with FIR delay stacking.

Usage:
    python test_narratives_local.py
"""
import sys
sys.path.insert(0, '/mnt/alphaidz/data')
sys.path.insert(0, '/mnt/alphaidz/litcoder_core')

import pickle
import numpy as np
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
                                                                                             

# --- Config ---
N_DELAYS = 4
N_FOLDS = 5
PKL_PATH = '/mnt/alphaidz/data/narratives_assembly_sub-249.pkl'
SUBJECT = 'sub-249'
LEAD_TRIM = 0
TRAIL_TRIM = 0


def load_narratives_from_pickle(pkl_path, subject=SUBJECT):
    """Convert litcoder_core pickle assembly to a NeuroidAssembly with word_info."""
    with open(pkl_path, 'rb') as f:
        assembly = pickle.load(f)

    all_brain_data = []
    all_stimuli = []
    all_story_ids = []
    all_stimulus_ids = []
    word_info = {}

    for story_name in assembly.stories:
        story = assembly.story_data[story_name]
        n_trs = story.brain_data.shape[0]
        si = story.split_indices

        start = LEAD_TRIM
        end = len(story.tr_times) - TRAIL_TRIM

        tr_stimuli = []
        for i in range(start, end):
            if i >= len(si):
                tr_stimuli.append('')
            elif i < len(si) - 1:
                words_in_tr = story.words[si[i]:si[i + 1]]
                tr_stimuli.append(' '.join(words_in_tr).strip())
            else:
                words_in_tr = story.words[si[i]:]
                tr_stimuli.append(' '.join(words_in_tr).strip())

        assert len(tr_stimuli) == n_trs, (
            f"Story {story_name}: {len(tr_stimuli)} stimuli vs {n_trs} brain TRs"
        )

        trimmed_tr_times = story.tr_times[start:end]

        tr_start_time = trimmed_tr_times[0]
        tr_end_time = trimmed_tr_times[-1] + np.mean(np.diff(trimmed_tr_times))
        word_mask = (np.array(story.data_times) >= tr_start_time) & \
                    (np.array(story.data_times) < tr_end_time)
        trimmed_words = [w for w, m in zip(story.words, word_mask) if m]
        trimmed_data_times = np.array(story.data_times)[word_mask]

        word_info[story_name] = {
            'words': trimmed_words,
            'data_times': trimmed_data_times.tolist(),
             'tr_times': list(trimmed_tr_times),
        }

        all_brain_data.append(story.brain_data)
        all_stimuli.extend(tr_stimuli)
        all_story_ids.extend([story_name] * n_trs)
        all_stimulus_ids.extend([f"{story_name}.{i}" for i in range(n_trs)])

    brain_data = np.vstack(all_brain_data)
    n_neuroids = brain_data.shape[1]

    neuroid_assembly = NeuroidAssembly(
        brain_data,
        dims=['presentation', 'neuroid'],
        coords={
            'stimulus': ('presentation', all_stimuli),
            'stimulus_id': ('presentation', all_stimulus_ids),
            'story': ('presentation', all_story_ids),
            'subject_id': ('neuroid', [subject] * n_neuroids),
            'neuroid_id': ('neuroid', [f"{subject}.{i}" for i in range(n_neuroids)]),
        }
    )
    neuroid_assembly.attrs['word_info'] = word_info
    return neuroid_assembly


# Override load_dataset to bypass S3
import brainscore_language
_original_load_dataset = brainscore_language.load_dataset

def _patched_load_dataset(identifier):
    if identifier == 'Narratives':
        return load_narratives_from_pickle(PKL_PATH)
    return _original_load_dataset(identifier)

brainscore_language.load_dataset = _patched_load_dataset

from brainscore_language.benchmarks.narratives import benchmark as narratives_benchmark_module
narratives_benchmark_module.load_dataset = _patched_load_dataset

from brainscore_language.benchmarks.narratives.benchmark import NarrativesRidge
benchmark = NarrativesRidge(n_folds=N_FOLDS)
print(f"Benchmark loaded: {benchmark.identifier}")
print(f"Data shape: {benchmark.data.shape}")
print(f"Stories: {sorted(set(benchmark.data['story'].values))}")
print(f"KFold splits: {benchmark.n_folds}")
print(f"word_info stories: {list(benchmark.data.attrs.get('word_info', {}).keys())}")

# --- Run benchmark ---
from brainscore_language import load_model
model = load_model('apertus-8b')

score = benchmark(model)
print(f"\n=== RESULTS ===")
print(f"Score: {score}")
print(f"Raw attributes: {score.attrs}")
