from brainscore_language import benchmark_registry
from .benchmark import LeBelLinear, LeBelRidge

benchmark_registry['LeBel-linear_pearsonr'] = LeBelLinear
benchmark_registry['LeBel-ridge_pearsonr'] = LeBelRidge
