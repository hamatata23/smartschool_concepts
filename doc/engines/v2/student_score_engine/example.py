"""Run from the project folder: python example.py."""
from engine import ScoreEngine, read_csv, train_with_holdout

model, report = train_with_holdout(read_csv("sample_students.csv"))
model.save("my_model.json")
print("Held-out MAE:", report["test"]["bounded_predictions"]["mae"])
student = {
    "study_hours_per_week": 15,
    "attendance_percent": 90,
    "previous_score": 75,
    "sleep_hours_per_night": 7,
    "assignments_completed_percent": 85,
}
loaded = ScoreEngine.load("my_model.json")
print("Prediction:", loaded.predict_details([student])[0])
print("Theta in original units:", loaded.coefficients())
