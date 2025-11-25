import os
import importlib.util
import pytest

spec = importlib.util.find_spec('torch')
if spec is None:
    pytest.skip("Torch not installed in the environment; skipping inference tests.", allow_module_level=True)

from predict import get_recommendation


def test_predict_artifacts_missing():
    missing_dir = os.path.join(os.path.dirname(__file__), 'nonexistent_artifacts')
    with pytest.raises(FileNotFoundError):
        get_recommendation(0, top_k=3, artifacts_dir=missing_dir)
