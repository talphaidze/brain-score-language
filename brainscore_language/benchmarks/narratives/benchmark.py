import numpy as np
import xarray as xr
from sklearn.model_selection import KFold
from tqdm import tqdm

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject


BIBTEX = """@article{nastase2021narratives,
  title={The "Narratives" fMRI dataset for evaluating models of naturalistic language comprehension},
  author={Nastase, Samuel A and Liu, Yun-Fei and Hillman, Hanna and Zadbood, Asieh and
          Hasenfratz, Liat and Keshavarzian, Neggin and Chen, Janice and Honey, Christopher J and
          Yeshurun, Yaara and Regev, Mor and others},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={250},
  year={2021},
  publisher={Nature Publishing Group}
}"""


def NarrativesLinear(n_folds=5):
    return _Narratives(metric='linear_pearsonr', n_folds=n_folds)


def NarrativesRidge(n_folds=5):
    return _Narratives(metric='ridge_pearsonr', n_folds=n_folds)


class _Narratives(BenchmarkBase):
    """
    Evaluate model ability to predict neural activity in response to the Narratives
    dataset (Nastase et al. 2021), recorded with fMRI (TR=1.5s).

    Uses KFold cross-validation (no shuffle) on TRs within each story,
    preserving temporal ordering to avoid autocorrelation leakage.

    Model embeddings are extracted per word and downsampled to TR rate using
    Lanczos interpolation, then FIR delay-stacked for hemodynamic response modeling.
    """

    def __init__(self, metric, n_folds=5):
        identifier = f"Narratives-{metric}"
        self.data = load_dataset('Narratives')
        self.metric = load_metric(metric)
        self.n_folds = n_folds

        super(_Narratives, self).__init__(
            identifier=identifier,
            version=1,
            parent='neural_language',
            ceiling=None,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI)

        stories = self.data['story'].values
        unique_stories = sorted(set(stories))
        # word_info = self.data.attrs.get('word_info', {})
        predictions = []

        for story in tqdm(unique_stories, desc='Narratives stories'):
            story_indexer = [s == story for s in stories]
            story_stimuli = self.data['stimulus'][story_indexer]

            # if story in word_info:
            #     words = word_info[story]['words']
            #     data_times = np.array(word_info[story]['data_times'])
            #     tr_times = np.array(word_info[story]['tr_times'])
            #     story_predictions = candidate.digest_text(
            #         words, data_times=data_times, tr_times=tr_times
            #     )['neural']
            # else:
            story_predictions = candidate.digest_text(story_stimuli.values)['neural']

            story_predictions['stimulus_id'] = 'presentation', story_stimuli['stimulus_id'].values
            story_predictions['story'] = 'presentation', story_stimuli['story'].values

            predictions.append(story_predictions)

        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self._kfold_score(predictions, self.data)

        # Aggregate across neuroids (voxels) to get a single score
        score = Score(raw_score.mean('neuroid').values)
        score.attrs['raw'] = raw_score

        if self.ceiling is not None:
            from brainscore_language.utils.ceiling import ceiling_normalize
            return ceiling_normalize(score, self.ceiling)
        return score

    def _kfold_score(self, predictions, target):
        """KFold CV (no shuffle) on TRs, averaged across folds."""
        n_trs = len(target['presentation'])
        kf = KFold(n_splits=self.n_folds, shuffle=False)

        fold_scores = []
        for fold_i, (train_idx, test_idx) in enumerate(
                tqdm(kf.split(np.arange(n_trs)), total=self.n_folds, desc='KFold CV')):
            train_source = predictions.isel(presentation=train_idx)
            train_target = target.isel(presentation=train_idx)
            test_source = predictions.isel(presentation=test_idx)
            test_target = target.isel(presentation=test_idx)

            fold_score = self.metric.apply(train_source, train_target, test_source, test_target)
            fold_scores.append(fold_score)

        # Average across folds
        fold_scores = Score(
            np.array([s.values for s in fold_scores]),
            dims=['split', *fold_scores[0].dims],
        )
        return fold_scores.mean('split')
