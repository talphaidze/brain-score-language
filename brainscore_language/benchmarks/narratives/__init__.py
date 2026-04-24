from brainscore_language import benchmark_registry
from .benchmark import NarrativesLinear, NarrativesRidge

benchmark_registry['Narratives-linear_pearsonr'] = NarrativesLinear
benchmark_registry['Narratives-ridge_pearsonr'] = NarrativesRidge
