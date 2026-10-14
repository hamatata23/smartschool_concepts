"""Multiple linear regression score predictor, Python 3.10+ and NumPy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

DEFAULT_FEATURES = ["study_hours_per_week", "attendance_percent", "previous_score",
                    "sleep_hours_per_night", "assignments_completed_percent"]


def numeric_matrix(rows, features):
    """Missing/blank values become NaN; malformed and infinite values are errors."""
    result = []
    for index, row in enumerate(rows, start=1):
        values = []
        for name in features:
            if name not in row:
                raise ValueError(f"Row {index}: missing feature column '{name}'")
            value = row[name]
            if value is None or (isinstance(value, str) and not value.strip()):
                values.append(np.nan)
                continue
            if isinstance(value, bool):
                raise ValueError(f"Row {index}: '{name}' must be a number, not boolean")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {index}: '{name}' is not numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"Row {index}: '{name}' must be finite or blank")
            values.append(value)
        result.append(values)
    return np.asarray(result, dtype=float).reshape(len(result), len(features))


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique column headers")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")
    if any(None in row for row in rows):
        raise ValueError("CSV contains more values than column headers")
    return rows


def targets(rows, target, score_min, score_max):
    y = numeric_matrix(rows, [target]).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("Target scores cannot be missing")
    if np.any((y < score_min) | (y > score_max)):
        raise ValueError(f"Target scores must be between {score_min} and {score_max}")
    return y


def metrics(actual, predicted):
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {"mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": None if denominator == 0 else float(1 - np.sum(residual ** 2) / denominator),
            "rows": len(actual)}


class ScoreEngine:
    """Numeric features, train-only median imputation and standardization.

    A fitted engine can be safely persisted as JSON (no executable pickle).
    Predictions are clipped to the configured score scale; raw values remain
    available through predict_details(). No uncertainty interval is claimed.
    """
    def __init__(self, features=None, score_min=0.0, score_max=100.0):
        self.features = list(DEFAULT_FEATURES if features is None else features)
        if not self.features or any(not isinstance(f, str) or not f.strip() for f in self.features):
            raise ValueError("Provide at least one nonempty feature name")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature names must be unique")
        if not np.isfinite([score_min, score_max]).all() or score_min >= score_max:
            raise ValueError("Score minimum must be finite and below maximum")
        self.score_min, self.score_max = float(score_min), float(score_max)
        self.fitted = False

    def fit(self, rows, target="final_score", solver="lstsq", l2=0.0,
            learning_rate=0.05, max_iterations=50000, tolerance=1e-9):
        self.fitted = False
        if target in self.features:
            raise ValueError("Target column must not also be a feature")
        if solver not in ("lstsq", "gd"):
            raise ValueError("Solver must be lstsq or gd")
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Iterations and tolerance must be positive")
        rows = list(rows)
        if len(rows) < 2:
            raise ValueError("Need at least two training rows")
        x = numeric_matrix(rows, self.features)
        y = targets(rows, target, self.score_min, self.score_max)
        if np.any(np.all(np.isnan(x), axis=0)):
            raise ValueError("A training feature has no observed values")
        self.medians = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.medians, x)
        self.means = np.mean(x, axis=0)
        self.scales = np.std(x, axis=0)
        self.scales[self.scales == 0] = 1.0
        self.minimums, self.maximums = np.min(x, axis=0), np.max(x, axis=0)
        design = np.column_stack([np.ones(len(x)), (x - self.means) / self.scales])
        rank = int(np.linalg.matrix_rank(design))
        # Objective: ||A theta-y||²/(2m) + l2*||theta[1:]||²/2.
        penalty = np.diag([0.0] + [1.0] * len(self.features))
        iterations, converged = 0, True
        if solver == "lstsq":
            # Augmented least squares avoids inversion, including singular inputs.
            a = np.vstack([design, np.sqrt(len(x) * l2) * penalty])
            b = np.concatenate([y, np.zeros(design.shape[1])])
            theta = np.linalg.lstsq(a, b, rcond=None)[0]
        else:
            hessian = design.T @ design / len(x) + l2 * penalty
            largest_eigenvalue = float(np.linalg.eigvalsh(hessian)[-1])
            if learning_rate >= 2.0 / largest_eigenvalue:
                raise ValueError(f"Learning rate too large; use less than {2/largest_eigenvalue:.6g}")
            theta = np.zeros(design.shape[1])
            converged = False
            for iterations in range(1, max_iterations + 1):
                gradient = design.T @ (design @ theta - y) / len(x) + l2 * (penalty @ theta)
                if np.linalg.norm(gradient, ord=np.inf) <= tolerance:
                    converged = True
                    break
                theta -= learning_rate * gradient
            if not np.isfinite(theta).all():
                raise ValueError("Optimization produced nonfinite parameters")
        self.theta = theta
        self.training = {"target": target, "rows": len(x), "solver": solver,
                         "l2": float(l2), "rank": rank, "columns": design.shape[1],
                         "iterations": iterations, "converged": converged}
        self.fitted = True
        return self

    def _check(self):
        if not self.fitted:
            raise ValueError("Train or load a model before prediction")

    def predict_details(self, rows):
        self._check()
        rows = list(rows)
        x = numeric_matrix(rows, self.features)
        missing = np.isnan(x)
        filled = np.where(missing, self.medians, x)
        outside = (filled < self.minimums) | (filled > self.maximums)
        standardized = (filled - self.means) / self.scales
        raw = self.theta[0] + standardized @ self.theta[1:]
        if not np.isfinite(raw).all():
            raise ValueError("Input magnitudes produced nonfinite predictions")
        clipped = np.clip(raw, self.score_min, self.score_max)
        return [{"predicted_score": float(clipped[i]), "raw_score": float(raw[i]),
                 "clipped": bool(raw[i] != clipped[i]),
                 "imputed_features": [f for j, f in enumerate(self.features) if missing[i, j]],
                 "outside_training_range": [f for j, f in enumerate(self.features) if outside[i, j]]}
                for i in range(len(rows))]

    def predict(self, rows):
        return np.array([r["predicted_score"] for r in self.predict_details(rows)])

    def coefficients(self):
        self._check()
        weights = self.theta[1:] / self.scales
        return {"intercept": float(self.theta[0] - self.means @ weights),
                "weights": dict(zip(self.features, map(float, weights)))}

    def evaluate(self, rows, target=None):
        rows = list(rows)
        self._check()
        if not rows:
            raise ValueError("Evaluation requires at least one row")
        y = targets(rows, target or self.training["target"], self.score_min, self.score_max)
        details = self.predict_details(rows)
        return {"bounded_predictions": metrics(y, np.array([d["predicted_score"] for d in details])),
                "raw_predictions": metrics(y, np.array([d["raw_score"] for d in details]))}

    def save(self, path):
        self._check()
        payload = {"format_version": 1, "features": self.features,
                   "score_min": self.score_min, "score_max": self.score_max,
                   "training": self.training}
        for key in ("medians", "means", "scales", "minimums", "maximums", "theta"):
            payload[key] = getattr(self, key).tolist()
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        try:
            model = cls(payload["features"], payload["score_min"], payload["score_max"])
            for key in ("medians", "means", "scales", "minimums", "maximums", "theta"):
                array = np.asarray(payload[key], dtype=float)
                expected = len(model.features) + (key == "theta")
                if array.shape != (expected,) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid model array: {key}")
                setattr(model, key, array)
            if np.any(model.scales <= 0) or np.any(model.minimums > model.maximums):
                raise ValueError("Invalid model scales or ranges")
            model.training = payload["training"]
            if not isinstance(model.training, dict) or not isinstance(model.training.get("target"), str):
                raise ValueError("Invalid training metadata")
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete or malformed model") from exc
        model.fitted = True
        return model


def train_with_holdout(rows, features=None, target="final_score", test_size=0.2, seed=42,
                       score_min=0, score_max=100, **fit_options):
    """Returns the training-only model and a held-out evaluation report.

    The exported model is NOT retrained on test rows, so the report evaluates
    exactly the saved model. Random splitting assumes independent student rows.
    """
    rows = list(rows)
    if len(rows) < 10:
        raise ValueError("Holdout training requires at least 10 rows")
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("Test size must be between zero and one")
    # Validate every row before any random split.
    schema = ScoreEngine(features, score_min, score_max)
    numeric_matrix(rows, schema.features)
    targets(rows, target, score_min, score_max)
    count = max(2, int(np.ceil(len(rows) * test_size)))
    if len(rows) - count < 2:
        raise ValueError("Test size leaves fewer than two training rows")
    indexes = np.random.default_rng(seed).permutation(len(rows))
    test = [rows[i] for i in indexes[:count]]
    train = [rows[i] for i in indexes[count:]]
    model = schema.fit(train, target=target, **fit_options)
    baseline = np.full(len(test), np.mean(targets(train, target, score_min, score_max)))
    warnings = []
    if model.training["rank"] < model.training["columns"]:
        warnings.append("Training design is rank deficient; individual coefficients may be unreliable.")
    if not model.training["converged"]:
        warnings.append("Gradient descent reached the iteration limit without convergence.")
    if len(train) <= len(model.features) + 1:
        warnings.append("Very few training rows relative to the number of parameters.")
    report = {"split": {"training_rows": len(train), "test_rows": len(test), "seed": seed},
              "training": model.training, "test": model.evaluate(test, target),
              "mean_baseline": metrics(targets(test, target, score_min, score_max), baseline),
              "coefficients_original_units": model.coefficients(), "warnings": warnings}
    return model, report


def generate_demo(path, count=500, seed=7):
    if count < 10:
        raise ValueError("Demo data requires at least 10 rows")
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(0, 30, count), rng.uniform(45, 100, count),
                         rng.uniform(25, 95, count), rng.uniform(4, 9, count),
                         rng.uniform(20, 100, count)])
    y = np.clip(3 + x @ np.array([0.7, 0.16, 0.45, 1.1, 0.12]) + rng.normal(0, 4, count), 0, 100)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(DEFAULT_FEATURES + ["final_score"])
        writer.writerows([[round(float(v), 3) for v in values] + [round(float(score), 3)]
                         for values, score in zip(x, y)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-data", help="Generate explicitly synthetic example data")
    demo.add_argument("--output", default="sample_students.csv")
    demo.add_argument("--rows", type=int, default=500)
    demo.add_argument("--seed", type=int, default=7)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="model.json")
    train.add_argument("--report", default="training_report.json")
    train.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    train.add_argument("--target", default="final_score")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--score-min", type=float, default=0)
    train.add_argument("--score-max", type=float, default=100)
    train.add_argument("--solver", choices=["lstsq", "gd"], default="lstsq")
    train.add_argument("--l2", type=float, default=0)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--max-iterations", type=int, default=50000)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--student", help="JSON object containing numeric features")
    source.add_argument("--input", help="CSV for batch prediction; extra columns are preserved")
    predict.add_argument("--output", help="Output CSV, required with --input")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--target")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo-data":
            generate_demo(args.output, args.rows, args.seed)
            print(f"Created {args.rows} SYNTHETIC rows in {args.output}")
        elif args.command == "train":
            paths = [Path(p).resolve() for p in (args.data, args.model, args.report)]
            if len(set(paths)) != 3:
                raise ValueError("Training data, model, and report paths must differ")
            model, report = train_with_holdout(read_csv(args.data), args.features, args.target,
                args.test_size, args.seed, args.score_min, args.score_max, solver=args.solver,
                l2=args.l2, learning_rate=args.learning_rate, max_iterations=args.max_iterations)
            model.save(args.model)
            Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "predict":
            model = ScoreEngine.load(args.model)
            if args.student is not None:
                student = json.loads(args.student)
                if not isinstance(student, dict):
                    raise ValueError("--student must be a JSON object")
                print(json.dumps(model.predict_details([student])[0], indent=2))
            else:
                if not args.output:
                    raise ValueError("--output is required for batch prediction")
                if Path(args.input).resolve() == Path(args.output).resolve():
                    raise ValueError("Prediction output must differ from input")
                rows = read_csv(args.input)
                details = model.predict_details(rows)
                columns = list(details[0])
                if set(columns).intersection(rows[0]):
                    raise ValueError("Input already has prediction columns; rename them first")
                with open(args.output, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + columns)
                    writer.writeheader()
                    for row, detail in zip(rows, details):
                        detail["imputed_features"] = ";".join(detail["imputed_features"])
                        detail["outside_training_range"] = ";".join(detail["outside_training_range"])
                        writer.writerow({**row, **detail})
                print(f"Wrote {len(rows)} predictions to {args.output}")
        else:
            report = ScoreEngine.load(args.model).evaluate(read_csv(args.data), args.target)
            print(json.dumps(report, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
