"""Run the learning/prediction core without any MySQL server."""
import json
from pathlib import Path

from demo_data import demo_rows
from learning import advance_state, normalize, predict_rows, utcnow
from settings import load_config

if __name__ == "__main__":
    c = load_config(Path(__file__).with_name("config.example.json"))
    now = utcnow()
    labeled, pending, _ = normalize(demo_rows(now), c, now)
    state, report = advance_state(None, labeled, c, now=now)
    if not state["active"]:
        raise RuntimeError(report)
    predictions = predict_rows(pending, state["active"], c)
    print(json.dumps({"synthetic_demo_only": True, "training_status": report["status"],
        "validation_mae": state["active"]["validation_mae"],
        "training_rows": state["active"]["training_rows"], "predictions": predictions}, indent=2))
