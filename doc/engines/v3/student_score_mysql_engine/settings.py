"""Strict configuration and literal .env parsing; no shell execution."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value):
        raise ValueError("SQL identifiers must be simple names of at most 64 characters")
    return "`" + value + "`"


def load_env(path):
    """Existing environment wins. No interpolation, export syntax, or shell parsing."""
    if path is None:
        return
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid .env entry on line {n}")
        os.environ.setdefault(key, value)


def validate(config):
    c = copy.deepcopy(config)
    required = {"pipeline_name", "state_dir", "source_table", "prediction_table", "columns",
                "features", "feature_ranges", "filter_equals", "score_min", "score_max", "max_rows", "training"}
    if set(c) != required:
        raise ValueError("Config keys differ from config.example.json")
    for k in ("pipeline_name", "source_table", "prediction_table"):
        identifier(c[k])
    if c["source_table"].lower() == c["prediction_table"].lower():
        raise ValueError("Source and prediction tables must be different")
    if not isinstance(c["state_dir"], str) or not c["state_dir"].strip():
        raise ValueError("state_dir must be a nonempty path")
    cols = c["columns"]
    if set(cols) != {"row_id", "student_id", "target_date", "features_as_of", "label_available_at", "target"}:
        raise ValueError("Configure all six source metadata columns")
    for name in cols.values():
        identifier(name)
    if len(set(x.lower() for x in cols.values())) != len(cols):
        raise ValueError("Metadata columns must be distinct")
    features = c["features"]
    if not isinstance(features, list) or not features:
        raise ValueError("Provide a nonempty feature list")
    for name in features:
        identifier(name)
    if len(set(f.lower() for f in features)) != len(features):
        raise ValueError("Features must be unique")
    if set(f.lower() for f in features) & set(x.lower() for x in cols.values()):
        raise ValueError("IDs, dates and target must not be features")
    if set(f.lower() for f in features) & {"row_id", "student_id", "target_date", "features_as_of", "label_available_at", "actual_score"}:
        raise ValueError("Feature names overlap reserved internal fields")
    if not isinstance(c["filter_equals"], dict):
        raise ValueError("filter_equals must be an object")
    for k, v in c["filter_equals"].items():
        identifier(k)
        if isinstance(v, (dict, list)) or (isinstance(v, float) and not math.isfinite(v)):
            raise ValueError("Filter values must be finite JSON scalars")
    def finite(v):
        return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    if not all(finite(c[k]) for k in ("score_min", "score_max")) or c["score_min"] >= c["score_max"]:
        raise ValueError("Invalid score bounds")
    if type(c["max_rows"]) is not int or c["max_rows"] < 2:
        raise ValueError("max_rows must be an integer >= 2")
    if not isinstance(c["feature_ranges"], dict) or set(c["feature_ranges"]) - set(features):
        raise ValueError("feature_ranges contains an unconfigured feature")
    for bounds in c["feature_ranges"].values():
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(finite(v) for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError("Each feature range needs two finite ordered numbers")
    t = c["training"]
    if set(t) != {"validation_fraction", "min_training_rows", "min_validation_rows", "max_validation_mae",
                  "require_better_than_mean_baseline", "l2", "solver", "learning_rate", "max_iterations", "tolerance"}:
        raise ValueError("Training keys differ from config.example.json")
    for k in ("min_training_rows", "min_validation_rows", "max_iterations"):
        if type(t[k]) is not int or t[k] < 2:
            raise ValueError(f"{k} must be an integer >= 2")
    for k in ("validation_fraction", "max_validation_mae", "l2", "learning_rate", "tolerance"):
        if not finite(t[k]):
            raise ValueError(f"{k} must be finite")
    if not 0 < t["validation_fraction"] < 1 or min(t["max_validation_mae"], t["l2"]) < 0:
        raise ValueError("Invalid validation fraction, MAE threshold or L2")
    if min(t["learning_rate"], t["tolerance"]) <= 0 or t["solver"] not in ("lstsq", "gd"):
        raise ValueError("Invalid solver settings")
    if type(t["require_better_than_mean_baseline"]) is not bool:
        raise ValueError("Baseline gate must be boolean")
    return c


def load_config(path):
    path = Path(path).resolve()
    c = validate(json.loads(path.read_text(encoding="utf-8")))
    c["state_dir"] = str((path.parent / c["state_dir"]).resolve())
    return c
