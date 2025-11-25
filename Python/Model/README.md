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
```
