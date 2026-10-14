"""Temporal validation and supervised retraining. This module never reads predictions."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import uuid

import numpy as np

from engine import ScoreEngine, metrics, numeric_matrix


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=str).encode()).hexdigest()


def contract(c):
    return {k: c[k] for k in ("pipeline_name", "source_table", "columns", "features",
                              "feature_ranges", "filter_equals", "score_min", "score_max")}


def timestamp(value, field):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time())
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp in {field}") from exc
    else:
        raise ValueError(f"Missing or invalid timestamp in {field}")
    if result.tzinfo:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def normalize(raw_rows, c, now=None):
    """Fail closed on invalid data; never quietly train on a reduced subset."""
    now = now or utcnow()
    cols = c["columns"]
    labeled, pending, seen = [], [], set()
    skipped = 0
    for i, raw in enumerate(raw_rows, 1):
        if set(cols.values()) - raw.keys():
            raise ValueError("Source row lacks a configured metadata column")
        row = {}
        for key in ("row_id", "student_id"):
            value = raw[cols[key]]
            if value is None or isinstance(value, (float, bool)) or not str(value).strip() or str(value) != str(value).strip() or len(str(value)) > 128:
                raise ValueError(f"Invalid {key} at source row position {i}")
            row[key] = str(value)
        if row["row_id"] in seen:
            raise ValueError("Source row IDs are not unique")
        seen.add(row["row_id"])
        exam = timestamp(raw[cols["target_date"]], "target_date")
        if exam.time() != time():
            raise ValueError("target_date must be a DATE, without a time component")
        as_of = timestamp(raw[cols["features_as_of"]], "features_as_of")
        if as_of >= exam:
            raise ValueError("features_as_of must precede the exam date")
        if as_of > now:
            raise ValueError("features_as_of cannot be in the future")
        row.update(target_date=exam.date().isoformat(), features_as_of=as_of.isoformat())
        values = numeric_matrix([raw], c["features"])[0]
        for feature, value in zip(c["features"], values):
            row[feature] = None if np.isnan(value) else float(value)
            bounds = c["feature_ranges"].get(feature)
            if bounds and row[feature] is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Feature {feature} is outside its allowed range at row position {i}")
        if np.isnan(values).all():
            raise ValueError(f"All features are missing at source row position {i}")
        score = raw[cols["target"]]
        available = raw[cols["label_available_at"]]
        if score is None:
            if available is not None:
                raise ValueError("Unknown scores must also have NULL score_recorded_at")
            row["actual_score"] = None
            row["label_available_at"] = None
            # A DATE has no exam time: do not issue forecasts on/after exam day.
            if exam.date() > now.date():
                pending.append(row)
            else:
                skipped += 1
        else:
            y = numeric_matrix([{"score": score}], ["score"])[0, 0]
            if not np.isfinite(y) or not c["score_min"] <= y <= c["score_max"]:
                raise ValueError("Actual score outside the configured scale")
            recorded = timestamp(available, "label_available_at")
            if recorded < exam or recorded > now or exam.date() > now.date():
                raise ValueError("Actual result timestamps must be after the exam and not in the future")
            row.update(actual_score=float(y), label_available_at=recorded.isoformat())
            labeled.append(row)
    return sorted(labeled, key=lambda r: r["row_id"]), sorted(pending, key=lambda r: r["row_id"]), skipped


def temporal_split(rows, c):
    t = c["training"]
    dates = sorted({r["target_date"] for r in rows})
    if len(dates) < 2:
        raise ValueError("Need at least two different historical exam dates for temporal validation")
    count = min(len(dates) - 1, max(1, math.ceil(len(dates) * t["validation_fraction"])))
    boundary = dates[-count]
    validation = [r for r in rows if r["target_date"] >= boundary]
    cutoff = min(r["features_as_of"] for r in validation)
    training = [r for r in rows if r["target_date"] < boundary and r["label_available_at"] <= cutoff]
    if len(training) < t["min_training_rows"] or len(validation) < t["min_validation_rows"]:
        raise ValueError(f"Insufficient temporal data: {len(training)} training and {len(validation)} validation rows")
    return training, validation, {"validation_start_exam_date": boundary,
        "training_label_cutoff_utc": cutoff, "training_rows": len(training),
        "validation_rows": len(validation), "purged_rows": len(rows)-len(training)-len(validation)}


def fit_model(rows, c):
    t = c["training"]
    options = {k: t[k] for k in ("solver", "l2", "learning_rate", "max_iterations", "tolerance")}
    return ScoreEngine(c["features"], c["score_min"], c["score_max"]).fit(rows, "actual_score", **options)


def train_candidate(rows, c, now=None):
    now = now or utcnow()
    training, validation, split = temporal_split(rows, c)
    candidate = fit_model(training, c)
    y = np.array([r["actual_score"] for r in validation])
    baseline = metrics(y, np.full(len(y), np.mean([r["actual_score"] for r in training])))
    measured = candidate.evaluate(validation, "actual_score")
    mae = measured["bounded_predictions"]["mae"]
    reasons = []
    if not candidate.training["converged"]:
        reasons.append("Solver did not converge")
    if mae > c["training"]["max_validation_mae"]:
        reasons.append("Validation MAE exceeds configured maximum")
    if c["training"]["require_better_than_mean_baseline"] and mae >= baseline["mae"]:
        reasons.append("Candidate does not beat the training-mean baseline")
    report = {"status": "rejected" if reasons else "accepted", "reasons": reasons,
              "split": split, "validation": measured, "mean_baseline": baseline,
              "candidate_training": candidate.training,
              "evaluation_scope": "Earlier-date fit evaluated on later exams; final model is refit on all labeled rows",
              "created_at_utc": now.isoformat()}
    if reasons:
        return None, report
    production = fit_model(rows, c)
    if not production.training["converged"]:
        report.update(status="rejected", reasons=["Full-data solver did not converge"])
        return None, report
    active = {"version": str(uuid.uuid4()), "contract_hash": digest(contract(c)),
              "model": production.to_dict(), "training_rows": len(rows),
              "latest_labeled_exam_date": max(r["target_date"] for r in rows),
              "trained_at_utc": now.isoformat(), "validation_mae": mae,
              "coefficients_original_units": production.coefficients(), "report": report}
    return active, report


def advance_state(state, rows, c, force=False, now=None):
    """Same labeled data + settings is a no-op, including a previous rejection."""
    now = now or utcnow()
    state = dict(state or {"format_version": 1, "active": None, "last_attempt": None})
    if state.get("format_version") != 1:
        raise ValueError("Unsupported state format")
    if state.get("active") and state["active"]["contract_hash"] != digest(contract(c)):
        state["active"] = None
    fingerprint = digest({"contract": contract(c), "training": c["training"], "labeled_rows": rows})
    if not force and (state.get("last_attempt") or {}).get("fingerprint") == fingerprint:
        return state, {"status": "unchanged", "last_training_status": state["last_attempt"]["report"]["status"]}
    try:
        active, report = train_candidate(rows, c, now)
    except ValueError as exc:
        active, report = None, {"status": "rejected", "reasons": [str(exc)], "created_at_utc": now.isoformat()}
    state["last_attempt"] = {"fingerprint": fingerprint, "report": report}
    if active:
        state["active"] = active
    return state, report


def predict_rows(pending, active, c):
    if active["contract_hash"] != digest(contract(c)):
        raise ValueError("Active model does not match current feature/target configuration")
    model = ScoreEngine.from_dict(active["model"])
    if model.features != c["features"]:
        raise ValueError("Stored model feature list differs from configuration")
    details = model.predict_details(pending)
    return [{"source_row_id": r["row_id"], "student_id": r["student_id"],
             "target_date": r["target_date"], "input_hash": digest(r), **d}
            for r, d in zip(pending, details)]
