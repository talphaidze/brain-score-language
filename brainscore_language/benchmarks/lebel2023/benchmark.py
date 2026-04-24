import numpy as np
import xarray as xr
from tqdm import tqdm

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.data.lebel2023 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize
from brainscore_language.benchmarks.lebel2023.ceiling import load_ceiling


CEILING_DEFAULTS = dict(
    trial_dir='/mnt/datasets/LeBel-UTS03',
    story_name='wheretheressmoke',
    subject='UTS03',
    lead_trim=40,
    n_splits=500,
    seed=42,
)


def LeBelLinear(test_story=None):
    return _LeBel(metric='linear_pearsonr',
                  test_story=test_story,
                  crossvalidation_kwargs=dict(
                      split_coord="story",
                      kfold="group",
                      random_state=1234
                  ))


def LeBelRidge(test_story=None):
    return _LeBel(metric='ridge_pearsonr',
                  test_story=test_story,
                  crossvalidation_kwargs=dict(
                      split_coord="story",
                      kfold="group",
                      random_state=1234
                  ))


class _LeBel(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in response to 25 naturalistic stories,
    recorded with fMRI (TR=2.0s) by LeBel et al. 2023.

    Alignment is evaluated via cross-validated linear or ridge predictivity,
    with story-level GroupKFold splits to prevent temporal leakage.

    Model embeddings are extracted per word and downsampled to TR rate using the
    specified method (default: Lanczos interpolation).

    Ceiling is computed via split-half reliability on repeated trial recordings.
    """

    def __init__(self, metric, test_story=None, crossvalidation_kwargs=None):
        identifier = f"LeBel-{metric}"
        self.data = load_dataset('LeBel.fROI')
        self.metric = load_metric(metric, crossvalidation_kwargs=crossvalidation_kwargs)
        self.test_story = test_story
        ceiling = load_ceiling(**CEILING_DEFAULTS)

        super(_LeBel, self).__init__(
            identifier=identifier,
            version=1,
            parent='neural_language',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI)

        stories = self.data['story'].values
        unique_stories = sorted(set(stories))
        word_info = self.data.attrs.get('word_info', {})
        predictions = []

        for story in tqdm(unique_stories, desc='LeBel stories'):
            story_indexer = [s == story for s in stories]
            story_stimuli = self.data['stimulus'][story_indexer]
            n_trs = story_stimuli.shape[0]
            
            # Pass TR-grouped stimuli (words joined per TR)
            # Lanczos within each TR is handled by output_to_representations.
            #story_predictions = candidate.digest_text(story_stimuli.values)['neural']

            if story in word_info:
                words = word_info[story]['words']
                data_times = np.array(word_info[story]['data_times'])
                tr_times = np.array(word_info[story]['tr_times'])
                story_predictions = candidate.digest_text(
                    words, data_times=data_times, tr_times=tr_times
                )['neural']
                story_predictions = story_predictions.isel(presentation=slice(10, len(tr_times) - 5))
            else:
                story_predictions = candidate.digest_text(story_stimuli.values)['neural']

            story_predictions['stimulus_id'] = 'presentation', story_stimuli['stimulus_id'].values
            story_predictions['story'] = 'presentation', story_stimuli['story'].values

            predictions.append(story_predictions)

        predictions = xr.concat(predictions, dim='presentation')

        import torch
        candidate.basemodel.cpu()
        torch.cuda.empty_cache()

        if self.test_story:
            # Fixed train/test split: train on all other stories, test on test_story
            raw_score = self._fixed_split_score(predictions, self.data)
        else:
            raw_score = self.metric(predictions, self.data)
        return ceiling_normalize(raw_score, self.ceiling)

    def _fixed_split_score(self, predictions, target):
        """Train on all stories except test_story, evaluate on test_story."""
        test_mask = target['story'].values == self.test_story
        train_mask = ~test_mask

        train_source = predictions.isel(presentation=train_mask)
        train_target = target.isel(presentation=train_mask)
        test_source = predictions.isel(presentation=test_mask)
        test_target = target.isel(presentation=test_mask)

        # # Extra trim on test set to match litcoder (skip first 40 TRs of test story)
        # test_source = test_source.isel(presentation=slice(30, None))
        # test_target = test_target.isel(presentation=slice(30, None))

        # Use the metric's regression and correlation directly
        score = self.metric.apply(train_source, train_target, test_source, test_target)
        return self.metric.aggregate(score)
