# Student Score Prediction Engine

Complete Python 3.10+ engine for multiple linear regression: train theta from
historical student records, evaluate on held-out students, persist a JSON model,
and predict individual students or a batch CSV. NumPy is the only dependency.

## Run immediately

Extract this project and open a terminal in its folder. On Windows use `py`
instead of `python` if needed.

```bash
python -m pip install -r requirements.txt
python engine.py train --data sample_students.csv --model model.json --report training_report.json
python engine.py predict --model model.json --input students_to_predict.csv --output predictions.csv
python example.py
python -m unittest discover -s tests -v
```

The included CSV, demo model, and report use **synthetic data** with a known
formula and random noise. They demonstrate functionality, not validated
accuracy on real students. Replace the training CSV with historical records
before using the engine for real predictions.

## Input

One CSV row per student. Default columns:

| Feature | Unit |
| --- | --- |
| study_hours_per_week | Hours per week |
| attendance_percent | Percent, 0–100 |
| previous_score | Prior comparable assessment, 0–100 |
| sleep_hours_per_night | Hours per night |
| assignments_completed_percent | Percent, 0–100 |
| final_score | Target score, 0–100; training/evaluation only |

Features must be numeric. Blank feature cells are filled with medians learned
only from training rows. In Python, use `None` for a missing observation.
Missing feature columns, malformed numbers, infinities, and fully blank
training features are rejected. Targets must be present and within the score
scale. Extra columns such as student identifiers are ignored in training and
preserved in batch output.

You can specify any numeric features and another score scale:

```bash
python engine.py train --data your_students.csv --features study_hours attendance prior_grade --target exam_score --score-max 20
```

Use `--score-max 20` only for a 0–20 target. The engine does not convert feature
units or encode text categories. Keep units consistent and validate domain
ranges such as attendance percent in your data collection process.

## Individual prediction

In bash:

```bash
python engine.py predict --model model.json --student '{"study_hours_per_week":15,"attendance_percent":90,"previous_score":75,"sleep_hours_per_night":7,"assignments_completed_percent":85}'
```

On Windows, use the batch CSV command above or `example.py` to avoid shell JSON
quoting differences. The output contains the bounded prediction, raw linear
score, whether it was clipped, imputed features, and features outside observed
training ranges. The outside-range flag indicates extrapolation; a combination
of values can still be unusual even if each is individually within range.
No confidence interval or probability is claimed.

## Python API

```python
from engine import ScoreEngine, read_csv, train_with_holdout

model, report = train_with_holdout(read_csv("sample_students.csv"), seed=42)
model.save("model.json")
student = {
    "study_hours_per_week": 15,
    "attendance_percent": 90,
    "previous_score": 75,
    "sleep_hours_per_night": 7,
    "assignments_completed_percent": 85,
}
print(model.predict([student]))
print(model.predict_details([student]))
print(model.coefficients())
loaded = ScoreEngine.load("model.json")
print(loaded.predict([student]))
```

`fit()` fits all provided rows. `train_with_holdout()` separates test students
first, learns preprocessing on training rows, and returns that training-only
model. It does not quietly retrain on test records. Models use JSON, not pickle.

## Mathematical derivation of theta

With p features, the model in original units is:

    predicted_score = theta_0 + theta_1*x_1 + ... + theta_p*x_p

Theta_0 is the intercept; the other theta values are feature weights. For m
rows, make a design matrix A with a leading column of ones and one column per
feature. Let y be the observed target vector. Predicted scores are A theta.

The squared-error cost and its derivative are:

    J(theta) = ||A theta - y||² / (2m)
    gradient J(theta) = Aᵀ (A theta - y) / m

Setting the gradient to zero yields the normal equations:

    Aᵀ A theta = Aᵀ y

If the design has full column rank:

    theta = (Aᵀ A)⁻¹ Aᵀ y

The implementation uses `numpy.linalg.lstsq`, avoiding explicit matrix
inversion. It also handles rank deficiency; when features are collinear,
individual coefficients may not be identifiable. The report flags that case.

Gradient descent updates all parameters simultaneously:

    theta_next = theta - learning_rate * Aᵀ(A theta - y) / m

The engine starts at zero and standardizes features using training means and
standard deviations: z_j = (x_j - mean_j) / scale_j. Constant columns use scale
1. Internal theta values apply to z rather than x. Original-unit coefficients
are recovered as:

    original_weight_j = standardized_weight_j / scale_j
    original_intercept = standardized_intercept - sum(mean_j * original_weight_j)

These are returned by `coefficients()` and included in the report.

```bash
python engine.py train --data sample_students.csv --solver gd --learning-rate 0.05 --model gd_model.json --report gd_report.json
```

Gradient descent checks the learning rate against the quadratic objective's
stable range and reports whether it converged. An iteration-limit warning
means the returned parameters have not met the convergence criterion; increase
iterations or use the default least-squares solver.

Optional L2 regularization leaves the intercept unpenalized:

    J(theta) = ||A theta-y||²/(2m) + lambda*||theta[1:]||²/2
    gradient = Aᵀ(A theta-y)/m + lambda*D theta
    D = diag(0, 1, ..., 1)

`--l2` is lambda, defaults to zero, and applies in standardized feature units.
Both solvers optimize the same objective. The least-squares path appends the
regularization rows to A, avoiding explicit inversion.

```bash
python engine.py train --data sample_students.csv --l2 0.1
```

Weights describe associations in the data, not causal effects of changing a
student's sleep, attendance, or study time.

## Evaluation and using real data

The default random split is 80% training / 20% testing with seed 42. Reports
include MAE, RMSE, and R² for raw and bounded predictions, plus a baseline that
predicts the training-set average for everyone. MAE and RMSE are in score
points and should be lower than that baseline. R² may be negative and is null
when evaluation target scores are constant.

```bash
python engine.py evaluate --model model.json --data held_out_students.csv
```

Random splitting assumes independent student rows. For repeated observations
of the same student, split by student externally. For a future-semester
forecast, split by time or cohort externally. Then:

```python
model = ScoreEngine().fit(read_csv("training_students.csv"))
print(model.evaluate(read_csv("future_cohort.csv")))
model.save("real_model.json")
```

Use only features available at prediction time. Avoid final exam outcomes or
other features that reveal the target. Keep an untouched final test set if
you tune features or regularization. Evaluate real records from the groups
and courses where you plan to use predictions. Changing curricula and linear
model assumptions may limit accuracy; no student's result is guaranteed.

## Included files

- `engine.py`: complete engine and command-line interface.
- `example.py`: Python API usage.
- `requirements.txt`: NumPy dependency.
- `sample_students.csv`: 500 synthetic labeled records.
- `students_to_predict.csv`: batch example including a missing observation.
- `demo_model.json`: saved synthetic model.
- `demo_training_report.json`: its held-out metrics and original-unit theta.
- `tests/test_engine.py`: 12 automated checks.

Generate fresh synthetic data with:

```bash
python engine.py demo-data --output sample_students.csv --rows 500 --seed 7
```
