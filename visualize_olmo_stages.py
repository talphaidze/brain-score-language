"""
Visualize per-voxel brain predictivity for different OLMo-3 training stages
projected onto the FSAverage5 cortical surface.

Usage:
    python visualize_olmo_stages.py --stage base
    python visualize_olmo_stages.py --stage sft
    python visualize_olmo_stages.py --stage dpo
    python visualize_olmo_stages.py --stage rlvr
    python visualize_olmo_stages.py --stage stage1-step10000  # early pre-training checkpoint

Available stages:
    base    - allenai/Olmo-3-1025-7B (base pre-trained)
    sft     - allenai/Olmo-3-7B-Instruct-SFT
    dpo     - allenai/Olmo-3-7B-Instruct-DPO
    rlvr    - allenai/Olmo-3-7B-Instruct (final RLVR)
    stage1-stepXXX - intermediate pre-training checkpoint (loaded via revision)
"""
import sys
sys.path.insert(0, '/mnt/alphaidz/litcoder_core')

import argparse
import os
import pickle
import numpy as np
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

sys.path.insert(0, '/mnt/alphaidz/brain-score-language')
from brain_plotting import BrainPlotter

# --- OLMo stage definitions ---
OLMO_STAGES = {
    'base': {
        'model_id': 'allenai/Olmo-3-1025-7B',
        'revision': None,
        'label': 'OLMo-3 7B Base',
    },
    'sft': {
        'model_id': 'allenai/Olmo-3-7B-Instruct-SFT',
        'revision': None,
        'label': 'OLMo-3 7B SFT',
    },
    'dpo': {
        'model_id': 'allenai/Olmo-3-7B-Instruct-DPO',
        'revision': None,
        'label': 'OLMo-3 7B DPO',
    },
    'rlvr': {
        'model_id': 'allenai/Olmo-3-7B-Instruct',
        'revision': None,
        'label': 'OLMo-3 7B RLVR',
    },
}

def get_stage_config(stage_name):
    """Get model config for a stage. Supports named stages and revision strings."""
    if stage_name in OLMO_STAGES:
        return OLMO_STAGES[stage_name]
    # Assume it's a revision string (e.g., stage1-step10000)
    return {
        'model_id': 'allenai/Olmo-3-1025-7B',
        'revision': stage_name,
        'label': f'OLMo-3 7B {stage_name}',
    }


def load_lebel_from_pickle(pkl_path, subject='UTS03', max_stories=None):
    """Convert litcoder_core pickle assembly to a NeuroidAssembly with word_info."""
    with open(pkl_path, 'rb') as f:
        assembly = pickle.load(f)

    all_brain_data = []
    all_stimuli = []
    all_story_ids = []
    all_stimulus_ids = []
    word_info = {}

    LEAD_TRIM = 10
    TRAIL_TRIM = 5

    stories = assembly.stories[:max_stories] if max_stories else assembly.stories

    for story_name in stories:
        story = assembly.story_data[story_name]
        n_trs = story.brain_data.shape[0]
        si = story.split_indices

        start = LEAD_TRIM
        end = len(story.tr_times) - TRAIL_TRIM

        tr_stimuli = []
        for i in range(start, end):
            if i < len(si) - 1:
                words_in_tr = story.words[si[i]:si[i + 1]]
            else:
                words_in_tr = story.words[si[i]:]
            text = ' '.join(words_in_tr).strip()
            tr_stimuli.append(text)

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

        # word_info[story_name] = {
        #     'words': trimmed_words,
        #     'data_times': trimmed_data_times.tolist(),
        #     'tr_times': trimmed_tr_times.tolist(),
        # }

        word_info[story_name] = {
            'words': list(story.words),
            'data_times': list(story.data_times),
            'tr_times': list(story.tr_times),
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


def load_olmo_model(stage_config, layer_idx=16):
    """Load an OLMo model at a specific training stage."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
    from brainscore_language.artificial_subject import ArtificialSubject

    model_id = stage_config['model_id']
    revision = stage_config['revision']

    print(f"Loading model: {model_id}" + (f" (revision: {revision})" if revision else ""))

    kwargs = {'trust_remote_code': True}
    if revision:
        kwargs['revision'] = revision

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return HuggingfaceSubject(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: [f'model.layers.{layer_idx}']
        },
    )


def visualize_scores(raw_scores, stage_label, output_dir='brain_olmo_stages'):
    """Plot brain surface maps using BrainPlotter and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    safe_label = stage_label.replace(' ', '_').lower()

    # All voxels are significant (no masking)
    sig_mask = np.ones(len(raw_scores), dtype=bool)

    plotter = BrainPlotter()
    fname = f'{output_dir}/{safe_label}_brain.png'
    fig = plotter.plot_left_right_hemisphere_correlations(
        correlations=raw_scores,
        significant_mask=sig_mask,
        title=stage_label,
    )
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    return [fname]


def main():
    parser = argparse.ArgumentParser(description='Visualize OLMo brain predictivity by training stage')
    parser.add_argument('--stage', type=str, required=True,
                        help='Training stage: base, sft, dpo, rlvr, or a revision string')
    parser.add_argument('--layer', type=int, default=16,
                        help='Layer index to use (default: 16, middle of 32-layer model)')
    parser.add_argument('--test-story', type=str, default='wheretheressmoke',
                        help='Test story for fixed split evaluation')
    parser.add_argument('--output-dir', type=str, default='brain_olmo_stages',
                        help='Output directory for brain images')
    parser.add_argument('--pkl-path', type=str, default='/mnt/alphaidz/data/assembly_lebel_uts03.pkl',
                        help='Path to LeBel pickle assembly')
    args = parser.parse_args()

    stage_config = get_stage_config(args.stage)
    print(f"\n=== {stage_config['label']} (layer {args.layer}) ===\n")

    # Patch load_dataset
    import brainscore_language
    _original_load_dataset = brainscore_language.load_dataset

    def _patched_load_dataset(identifier):
        if identifier == 'LeBel.fROI':
            return load_lebel_from_pickle(args.pkl_path)
        return _original_load_dataset(identifier)

    brainscore_language.load_dataset = _patched_load_dataset
    from brainscore_language.benchmarks.lebel2023 import benchmark as lebel_benchmark_module
    lebel_benchmark_module.load_dataset = _patched_load_dataset

    # Load benchmark and model
    from brainscore_language.benchmarks.lebel2023.benchmark import LeBelRidge
    benchmark = LeBelRidge(test_story=args.test_story)
    model = load_olmo_model(stage_config, layer_idx=args.layer)

    print(f"Benchmark: {benchmark.identifier}")
    print(f"Data shape: {benchmark.data.shape}")

    # Run benchmark
    score = benchmark(model)
    if isinstance(score, dict):
        best_layer = max(score, key=lambda k: score[k].values)
        print(f"Best layer: {best_layer}")
        score = score[best_layer]

    print(f"\nScore (ceiling-normalized): {score.values:.4f}")

    # Extract per-voxel scores
    raw_scores = score.attrs['raw']
    if hasattr(raw_scores, 'attrs') and 'raw' in raw_scores.attrs:
        # Get per-voxel raw scores (the neuroid-level array)
        per_voxel = raw_scores.attrs['raw']
        if hasattr(per_voxel, 'values'):
            per_voxel = per_voxel.values
    elif hasattr(raw_scores, 'values'):
        per_voxel = raw_scores.values
    else:
        per_voxel = np.array(raw_scores)
    per_voxel = np.array(per_voxel).flatten()

    print(f"Per-voxel scores shape: {per_voxel.shape}")
    print(f"Score range: [{per_voxel.min():.4f}, {per_voxel.max():.4f}]")

    # Visualize
    stage_label = f"{stage_config['label']} (L{args.layer})"
    visualize_scores(per_voxel, stage_label, output_dir=args.output_dir)

    # Save raw scores for later comparison
    np.save(f"{args.output_dir}/{args.stage}_layer{args.layer}_voxel_scores.npy", per_voxel)
    print(f"\nSaved voxel scores to {args.output_dir}/{args.stage}_layer{args.layer}_voxel_scores.npy")


if __name__ == '__main__':
    main()
