# Q3: load the pickled pipeline and score the first lead
import pickle
from model import normalize_record

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

proba = pipeline.predict_proba([normalize_record(lead)])[0, 1]
print("Conversion probability:", round(float(proba), 3))
