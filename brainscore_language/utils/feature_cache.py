import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "brainscore_language" / "features"


class FeatureCache:
    """Caches raw model activations (NeuroidAssembly) to disk.

    Keyed by model_id + input stimuli text, so that changing downstream
    processing (FIR, downsampling, metrics) reuses the same cached features.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_cache_key(model_id: str, text: list, layers: list) -> str:
        params = {
            "model_id": model_id,
            "text": text,
            "layers": sorted(layers),
        }
        key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return key

    def get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def save(self, model_id: str, text: list, layers: list, neural_assembly):
        cache_key = self._make_cache_key(model_id, text, layers)
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, "wb") as f:
            pickle.dump(neural_assembly, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cached features to {cache_path}")
        print(f"Cached features to {cache_path}")

    def load(self, model_id: str, text: list, layers: list):
        cache_key = self._make_cache_key(model_id, text, layers)
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading cached features from {cache_path}")
            print(f"Loading cached features from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None
