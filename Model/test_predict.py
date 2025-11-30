import os
import importlib.util
import pytest

spec = importlib.util.find_spec('torch')
if spec is None:
    pytest.skip("Torch not installed in the environment; skipping inference tests.", allow_module_level=True)

from Model.predict import get_recommendation, map_external_id_to_traveler_name


def test_predict_artifacts_missing():
    missing_dir = os.path.join(os.path.dirname(__file__), 'nonexistent_artifacts')
    with pytest.raises(FileNotFoundError):
        get_recommendation(0, top_k=3, artifacts_dir=missing_dir)


def test_map_external_user_id(tmp_path):
    # Use a local CSV file in the Data folder for deterministic behavior
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'users_generated.csv'))
    # pick a known user id value from the sample 'users_generated.csv' in Data
    # The dataset includes U0001, ensure the file exists
    assert os.path.exists(csv_path)
    name = map_external_id_to_traveler_name('U0001', csv_path=csv_path)
    assert isinstance(name, str) and len(name) > 0
