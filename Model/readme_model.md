# Model inference helper

Use `get_recommendation(user_identifier)` to fetch recommended destinations for a user.

Usage examples:

python (CLI example):
```powershell
cd Projet_Rec_Sys\Python\Model
python -c "from predict import get_recommendation; print(get_recommendation(0, top_k=5))"
```

In a web application, prefer to instantiate a `Predictor` at startup so the model and graph are loaded once:

```python
from Python.Model.predict import Predictor
predictor = Predictor()
predictor.load()  # heavy operation
recs = predictor.recommend(0, top_k=5)
```

Make sure to install dependencies from `requirements.txt` before running:

```powershell
pip install -r requirements.txt

Evaluation (NDCG@k)
--------------------
You can run an offline NDCG@k evaluation using the saved model artifacts and a local CSV dataset (for instance the synthetic dataset in the project root):

```powershell
python -m Model.eval --artifacts_dir Model/artifacts --csv_path ..\..\synthetic_travel_data_daily_cost_coherent.csv --k 10
```

The eval script will align the CSV with the saved mappings, drop rows that cannot be mapped to saved categories, and compute a leave-one-out NDCG@k for the remaining users.

Plotting training metrics
-------------------------
If you ran training via `python -m Model.train` and metrics were saved to the artifacts folder, plot them with:

```powershell
python -m Model.plot_metrics --artifacts_dir Model/artifacts --output Model/artifacts/metrics.png
```

This will create `metrics.png` under `Model/artifacts/` showing training loss, validation loss, and Val NDCG@10 across epochs.
```
