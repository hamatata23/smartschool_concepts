# Student score prediction engine — MySQL edition

Python 3.10+ • MySQL 8.0.22+ / 8.4 • Linux • NumPy + PyMySQL

Reads historical results and upcoming exam records from `tbl_student_score`, learns
multiple linear regression weights, and writes forecasts to `tbl_student_prediction_score`.
It runs locally and does not send student data to an AI service. After installing
dependencies, it needs only a connection to your MySQL server, including a local LAN.

**This is supervised learning. Teachers must supply real exam scores.** It cannot
learn correct answers from unknown scores, and it never trains on its own forecasts.
Automatic retraining does not guarantee increasing accuracy.

## 1. What you receive

| File | Purpose |
|---|---|
| `mysql_engine.py` | MySQL command-line entry point |
| `engine.py` | Regression, gradient descent, median imputation, model JSON |
| `learning.py` | Data checks, temporal validation, retraining decisions |
| `database.py` | MySQL reads, transactions, locks, prediction upserts |
| `settings.py` | Configuration and credential loading |
| `config.example.json` | Column names, numeric features, cohort filters, quality thresholds |
| `.env.example` | Database connection settings |
| `sql/` | Source example and required destination schema |
| `deploy/` | Hourly systemd timer and service |
| `example_offline.py`, `demo_data.py` | Synthetic demonstration; no real student data |
| `tests/` | Unit/workflow tests and an optional real MySQL integration test |
| `docs/REGRESSION.md` | Meaning and mathematical derivation of theta |
| `docs/OPERATIONS.md` | Data contract, limitations, recovery and troubleshooting |

## 2. Your source data must represent a forecast task

One row means **one student, one subject/grade, one exam, and one fixed pre-exam
feature snapshot**. A student can have many rows across years. Do not overwrite
last year's row when creating next year's forecast. Use a new unique row ID.

| Default column | Meaning |
|---|---|
| `id` | Unique, stable source record ID; primary or single-column unique index |
| `student_id` | Student identifier; never used as a regression input |
| `subject`, `grade_level` | Cohort filters; sample config selects mathematics, grade 8 |
| `exam_date` | DATE of the exam being predicted |
| `features_as_of` | UTC DATETIME when this feature snapshot was available; before exam day |
| Five configured feature columns | Numeric input measurements, with consistent units |
| `final_score` | Actual score, 0–100 by default; **NULL when unknown** |
| `score_recorded_at` | UTC DATETIME when the actual score became available; NULL while unknown |

Historical feature snapshots must actually contain only information available at
`features_as_of`. Adding a timestamp to end-of-year attendance does not make that
attendance a valid beginning-of-year predictor. For next-year forecasts, build
historical pairs of prior-year inputs and the following year's actual score, using
a comparable forecast horizon throughout.

Default inputs: `study_hours_per_week`, `attendance_percent`, `previous_score`,
`sleep_hours_per_night`, `assignments_completed_percent`.

Use SQL NULL for an individually missing feature. The model fills it with that
feature's training median. All features missing in a row, or a completely missing
training feature, is an error. A real score of 0 is valid and is not "unknown".

**Existing database:** map names in `config.json`. Do not blindly replace or run
the example source schema over your production design. The engine does not ALTER
your source table. If historical snapshots/dates are missing, prepare them first.

**New/demo database:** `sql/example_source.sql` defines a compatible table.
The required prediction schema is `sql/prediction_table.sql`. An existing output
table with a different schema needs a reviewed migration; the engine stops rather
than automatically altering it.

## 3. Install on Ubuntu/Debian

Upload and extract `student_score_mysql_engine.zip`, then:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip default-mysql-client
cd student_score_mysql_engine
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.json config.json
cp .env.example .env
chmod 600 .env
```

Python must be 3.10 or newer. Edit `.env` with your database credentials and edit
`config.json` to match your columns, cohort and score scale. `.env` values are
literal: do not put quotes around passwords, use `export`, or add inline comments.
Existing process environment variables override this file. Do not share `.env`.

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=school
MYSQL_USER=score_engine
MYSQL_PASSWORD=your_actual_password
```

For a remote MySQL host, also configure `MYSQL_SSL_CA` with its CA certificate.
Certificate and hostname verification are enabled. Local sockets are supported
through `MYSQL_UNIX_SOCKET`.

Use a database administrator to create the destination table once:

```bash
mysql -u YOUR_DB_ADMIN -p school < sql/prediction_table.sql
```

The SQL file uses the default output name; adjust it if you changed that name in
config. Runtime credentials need SELECT on the source table and SELECT, INSERT,
UPDATE on the prediction table. They do not need write access to student records.

Example grants, after creating the user separately (adjust the account's host):

```sql
GRANT SELECT ON school.tbl_student_score TO 'score_engine'@'localhost';
GRANT SELECT, INSERT, UPDATE ON school.tbl_student_prediction_score TO 'score_engine'@'localhost';
```

Alternatively, an account with CREATE permission can use
`python mysql_engine.py init-output --config config.json --env-file .env`.
It creates the output if missing and checks both schemas. It does not create the source.

## 4. Check and run

```bash
python -m unittest discover -s tests -v
python example_offline.py
python mysql_engine.py check --config config.json --env-file .env
python mysql_engine.py run-once --config config.json --env-file .env
```

The offline example uses 400 synthetic historical rows and five upcoming records.
Its example MAE is about 1.44 points, and its first prediction is about 67.91.
**Those numbers say nothing about real-world accuracy.**

`check` validates connectivity, schemas and source rows without writing predictions.
`run-once` trains if needed, saves the model, updates predictions, and exits.
If no model passes validation, it exits with an error and writes the reason to
`var/math_grade8/last_run.json`. It does not fabricate initial predictions.

Read current predictions:

```sql
SELECT source_row_id, student_id, target_date, predicted_score,
       model_version, validation_mae, warnings_json, predicted_at
FROM tbl_student_prediction_score
WHERE pipeline_name = 'school_math_grade8'
  AND is_current = 1
ORDER BY student_id, target_date;
```

`validation_mae` describes the model's validation set, not confidence in one
student's prediction. `warnings_json` identifies imputed inputs, values beyond
the training range, and clipping to the configured score scale.

## 5. How automatic learning works

Every run:

1. Reads configured source fields for the selected cohort.
2. Hashes historical actual scores, feature snapshots, dates and training settings.
3. If that training content changed, fits a candidate model. New, corrected and
   deleted historical records are detected, even without an `updated_at` column.
4. Validates on the newest exam-date groups. Earlier training results must have
   been available before the earliest validation feature snapshot.
5. Accepts the candidate only if it passes the configured MAE threshold and, by
   default, beats a training-mean baseline. Then refits on all historical labels.
6. Otherwise retains the previously accepted compatible model. A rejected
   snapshot is not retried until data/settings change or `--force-retrain` is used.
7. Re-reads upcoming records and writes predictions in one transaction.

Default gates: at least 50 training rows, 20 validation rows, at least two distinct
historical exam dates, newest 20% of distinct dates used for validation, maximum
MAE 12 points on a 0–100 scale. Purging late results can reduce the training set.
These are initial operating settings, not a guarantee that 70 records are enough.
Choose acceptable error based on your real use case.

Changing an upcoming student's inputs changes the next prediction. It does not
create a new training answer. Once the real result is known, your school app can
record it, using the correct ID and actual availability time:

```sql
UPDATE tbl_student_score
SET final_score = 83,
    score_recorded_at = UTC_TIMESTAMP()
WHERE id = 123;
```

Do this only after that exam. Leave its historical pre-exam feature snapshot intact.
The next scheduled run includes the actual result in its training review.

## 6. Enable unattended hourly runs

The included service assumes installation at `/opt/student_score_mysql_engine`.
Use the following after reviewing config and confirming a successful manual run.
If moving an already-installed project, recreate the virtual environment at its
final location; virtual environments are not reliably relocatable.

```bash
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin scoreengine
sudo mkdir -p /opt/student_score_mysql_engine
sudo cp -a . /opt/student_score_mysql_engine/
cd /opt/student_score_mysql_engine
sudo python3 -m venv --clear .venv
sudo .venv/bin/python -m pip install -r requirements.txt
sudo mkdir -p var
sudo chown -R root:scoreengine /opt/student_score_mysql_engine
sudo chmod -R o-rwx /opt/student_score_mysql_engine
sudo chmod 640 .env
sudo chown -R scoreengine:scoreengine var
sudo cp deploy/student-score.service deploy/student-score.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start student-score.service
sudo systemctl enable --now student-score.timer
```

Skip `useradd` if the service account already exists. Keep config `state_dir`
under `/opt/student_score_mysql_engine/var`, as allowed by the service sandbox.
Make any TLS CA file readable by the service user.

```bash
systemctl list-timers student-score.timer
sudo journalctl -u student-score.service -n 100 --no-pager
```

The timer checks hourly with up to one minute of jitter. No application port or
web server is needed. Data/model files stay on your server. The timer is included
in this download; it has not been installed on your server by ChatGPT.

## 7. Add more inputs

1. Add numeric columns to the source and populate truthful historical snapshots.
2. Add their names to `features` and optionally their bounds to `feature_ranges`.
3. Run `check`, then `run-once`. A feature change requires a new accepted model.

For example, add `quiz_average` and `tutoring_hours_per_week`. Do not add names,
student IDs, the target exam result, or columns measured after the prediction date.
Categorical fields require deliberate encoding; this edition does not infer it.

Use separate config files, `pipeline_name` values and state directories for
different subjects/grades or incompatible exam scales. Configure separate services
for them. Do not mix heterogeneous curricula into one model without evaluating it.

## 8. Test status and real MySQL verification

At packaging: **35 automated tests passed; one optional live MySQL integration
test was skipped** because no MySQL server was available in the build environment.
Unit/workflow tests cover regression, temporal leakage checks, retraining triggers,
rejected candidates, persistence, SQL binding, concurrent-run locks and rollback
paths. Database calls in workflow tests are mocked; this does not replace testing
against your server and schema.

The integration test creates randomly named test tables, exercises training,
upserts, repeat runs, feedback and completed exams, then drops only those tables.
Run it only in a disposable database ending in `_engine_test`:

```bash
MYSQL_HOST=127.0.0.1 MYSQL_DATABASE=school_engine_test \
MYSQL_USER=test_user MYSQL_PASSWORD='your_test_password' \
RUN_MYSQL_INTEGRATION=1 python -m unittest discover -s tests -p test_live_mysql.py -v
```

For an optional full demo, create an empty disposable database, apply
`sql/example_source.sql` and `sql/prediction_table.sql`, then:

```bash
python demo_data.py --output demo_students.sql
mysql -u YOUR_DB_ADMIN -p school_engine_test < demo_students.sql
```

Point `.env` at that database and run the normal commands. Do not import synthetic
demo data into real training tables. MariaDB compatibility is not claimed.

## 9. Install in an area without Internet

On an Internet-connected machine with the **same Linux architecture and Python
version** as your server:

```bash
python -m pip download --only-binary=:all: -r requirements.txt -d wheelhouse
```

Transfer the project and `wheelhouse` to the server, then install inside its venv:

```bash
python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
```

Python, venv support and MySQL must already be installed using offline OS packages
if necessary. Record tested dependency versions with `python -m pip freeze` after
deployment. Keep dependency wheels for reproducible reinstallations.

Further details: [operations](docs/OPERATIONS.md) and [theta derivation](docs/REGRESSION.md).
