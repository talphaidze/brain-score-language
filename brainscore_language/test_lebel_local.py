"""
Test LeBel benchmark locally using the pickle assembly from litcoder_core,
bypassing S3.

Usage:
    python test_lebel_local.py
"""
import sys
sys.path.insert(0, '/mnt/alphaidz/litcoder_core')

import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_language import data_registry
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils.ceiling import ceiling_normalize

#test story variable for wheresthesmoke instead of trimming first 5 trim the first 40 in the brain data (test set)
def load_lebel_from_pickle(pkl_path, subject='UTS03', max_stories=None):
    """Convert litcoder_core pickle assembly to a NeuroidAssembly."""
    with open(pkl_path, 'rb') as f:
        assembly = pickle.load(f)

    all_brain_data = []
    all_stimuli = []
    all_story_ids = []
    all_stimulus_ids = []
    word_info = {}

    # Each story's brain_data has 15 TRs trimmed vs tr_times/split_indices:
    # 5 pre-stimulus baseline + 10 trailing TRs.
    # how we figured this out: each stories' tr times [:10] has first five as - (seconds before story beings),
    # meaning it's baseline
    # then, tr_times = brain_data trs + 15, so 15-5=10 trailing trs

    LEAD_TRIM = 10 # we need to remove addiotional 5 because there's noise
    TRAIL_TRIM = 5 # and from the end we only remove 5

    stories = assembly.stories[:max_stories] if max_stories else assembly.stories

    for story_name in stories:
        story = assembly.story_data[story_name]
        n_trs = story.brain_data.shape[0]
        si = story.split_indices #how many words have been spoken at that tr point

        # Build TR-level stimuli by joining words within each TR window
        start = LEAD_TRIM
        end = len(story.tr_times) - TRAIL_TRIM  # should equal LEAD_TRIM + n_trs

        tr_stimuli = []
        for i in range(start, end):
            if i < len(si) - 1:
                words_in_tr = story.words[si[i]:si[i + 1]]
            else:
                words_in_tr = story.words[si[i]:]
                # this is commented out because we wil handle empty trs directly in digest_text
            text = ' '.join(words_in_tr).strip()
            # if text == '': # in hf wrapper a condition that if there's an empty text just append 0s to the features
            #     text = '.'  # placeholder for silent TRs to avoid empty tokenization
            tr_stimuli.append(text)

        assert len(tr_stimuli) == n_trs, (
            f"Story {story_name}: {len(tr_stimuli)} stimuli vs {n_trs} brain TRs"
        )

        # Trim tr_times and data_times to match brain_data
        trimmed_tr_times = story.tr_times[start:end]

        # Filter words that fall within the trimmed TR window
        tr_start_time = trimmed_tr_times[0]
        tr_end_time = trimmed_tr_times[-1] + np.mean(np.diff(trimmed_tr_times))
        word_mask = (np.array(story.data_times) >= tr_start_time) & \
                    (np.array(story.data_times) < tr_end_time)
        trimmed_words = [w for w, m in zip(story.words, word_mask) if m]
        trimmed_data_times = np.array(story.data_times)[word_mask]

        # blind shot
        word_info[story_name] = {                                                                     
            'words': list(story.words),                                                             
            'data_times': list(story.data_times),                                                     
            'tr_times': list(story.tr_times),                                                         
        }

        # word_info[story_name] = {
        #     'words': trimmed_words,
        #     'data_times': trimmed_data_times.tolist(),
        #     'tr_times': trimmed_tr_times.tolist(),
        # }

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
    # Attach word_info so the benchmark uses the word-level + downsampling path
    neuroid_assembly.attrs['word_info'] = word_info
    return neuroid_assembly


# --- Config ---
MAX_STORIES = None  # Set to None for all 25 stories
N_DELAYS=4 # this is deprecated from before when we had fri inside the benchmark
TEST_STORY = 'wheretheressmoke'

# Override load_dataset to bypass S3 for LeBel
PKL_PATH = '/mnt/alphaidz/data/assembly_lebel_uts03.pkl'

import brainscore_language
_original_load_dataset = brainscore_language.load_dataset

def _patched_load_dataset(identifier):
    if identifier == 'LeBel.fROI':
        return load_lebel_from_pickle(PKL_PATH, max_stories=MAX_STORIES)
    return _original_load_dataset(identifier)

brainscore_language.load_dataset = _patched_load_dataset

# Patch the module-level import in the benchmark module too
from brainscore_language.benchmarks.lebel2023 import benchmark as lebel_benchmark_module
lebel_benchmark_module.load_dataset = _patched_load_dataset

from brainscore_language.benchmarks.lebel2023.benchmark import LeBelRidge
benchmark = LeBelRidge(test_story=TEST_STORY)
print(f"Benchmark loaded: {benchmark.identifier}")
print(f"Data shape: {benchmark.data.shape}")
print(f"Stories: {sorted(set(benchmark.data['story'].values))}")
print(f"First 5 stimuli: {benchmark.data['stimulus'].values[:5]}")

# --- Run model and diagnose ---
from brainscore_language import load_model
model = load_model('apertus-8b')

# The benchmark.__call__ handles word-level extraction + Lanczos downsampling
# when word_info is present in the data attrs
score = benchmark(model)
if isinstance(score, dict):
    score = next(iter(score.values()))
print(f"\n=== RESULTS ===")
print(f"Score: {score}")
print(f"Raw attributes: {score.attrs}")

import pickle
raw_score = score.attrs['raw']
per_voxel_corrs = raw_score.attrs['raw'].values
neuroid_ids = raw_score.attrs['raw']['neuroid_id'].values

results = {
      'mean_score': float(raw_score.values),
      'median_score': float(np.median(per_voxel_corrs)),
      'correlations': per_voxel_corrs,
      'neuroid_ids': neuroid_ids,
      'ceiling_normalized_score': float(score.values),
      'ceiling': float(score.attrs['ceiling'].values),
  }

out_path = 'brainscore_results_layer9_sum.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {out_path}")

# model.start_neural_recording(
#     recording_target=ArtificialSubject.RecordingTarget.language_system,
#     recording_type=ArtificialSubject.RecordingType.fMRI)

# stimuli = benchmark.data['stimulus']
# stories = benchmark.data['story'].values
# predictions = []
# for story in tqdm(sorted(set(stories)), desc='LeBel stories'):
#     story_indexer = [s == story for s in stories]
#     story_stimuli = stimuli[story_indexer]
#     story_predictions = model.digest_text(story_stimuli.values)['neural']
#     story_predictions['stimulus_id'] = 'presentation', story_stimuli['stimulus_id'].values
#     story_predictions['story'] = 'presentation', story_stimuli['story'].values
#     predictions.append(story_predictions)

# predictions = xr.concat(predictions, dim='presentation')

# print("\n=== DIAGNOSTICS ===")
# print(f"Predictions shape: {predictions.shape}")
# print(f"Predictions coords: {list(predictions.coords)}")
# print(f"Predictions dims: {predictions.dims}")
# print(f"Brain data shape: {benchmark.data.shape}")
# print(f"Brain data coords: {list(benchmark.data.coords)}")

# # Check NaN in predictions
# pred_nans = np.isnan(predictions.values).sum()
# print(f"\nNaN in predictions: {pred_nans}")
# if pred_nans > 0:
#     nan_rows = np.where(np.isnan(predictions.values).any(axis=1))[0]
#     print(f"NaN row indices: {nan_rows[:20]}")
#     print(f"stimulus_id at NaN rows: {predictions['stimulus_id'].values[nan_rows[:10]]}")

# # Check NaN in brain data
# brain_nans = np.isnan(benchmark.data.values).sum()
# print(f"NaN in brain data: {brain_nans}")

# # Check stimulus_id alignment
# pred_ids = sorted(predictions['stimulus_id'].values)
# brain_ids = sorted(benchmark.data['stimulus_id'].values)
# print(f"\nstimulus_id match: {pred_ids == brain_ids}")
# print(f"Prediction stimulus_ids (first 5): {predictions['stimulus_id'].values[:5]}")
# print(f"Brain data stimulus_ids (first 5): {benchmark.data['stimulus_id'].values[:5]}")



# # Try the metric manually
# if pred_nans == 0 and brain_nans == 0:
#     print("\nNo NaN detected — running metric...")
#     raw_score = benchmark.metric(predictions, benchmark.data)
#     score = ceiling_normalize(raw_score, benchmark.ceiling)
#     print(f"Score: {raw_score}")
#     print(f"Ceiling-normalized score: {score}")
# else:
#     print("\nNaN detected — skipping metric. Fix NaN source first.")

