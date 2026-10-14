# StudentAbility AI | MySQL + MongoDB

**Working reference implementation (v1.0)** of a Bayesian 2-parameter logistic item-response-theory (IRT) ability assessment engine for **Maths, Physics, English, Chemistry and Biology**.

It contains one shared psychometric and REST API codebase and **two database adapters**: SQLAlchemy/PyMySQL for MySQL, and PyMongo for MongoDB. The backends are interchangeable via `DB_BACKEND`. They do not automatically synchronize data.

**Scope:** correct/incorrect multiple-choice questions; 3 example topics per subject; 60 seed questions; teacher/admin item authoring; adaptive item selection; ability reports with approximate uncertainty; optional offline item recalibration; persisted estimate snapshots. The sample question-bank calibration values and generated learners are **synthetic and unvalidated**. This is a deployable engineering prototype, **not** a clinically/psychometrically validated test.

## 1. What you get

| Capability | Implementation |
| --- | --- |
| Five subjects | Maths, Physics, English, Chemistry, Biology |
| 15 topic categories | Three examples per subject; edit taxonomy in `app/schemas.py` |
| Student model | Bayesian MAP ability (2PL IRT) and Laplace 95% intervals |
| Assessment | Server-side correctness scoring; one response per student/item |
| Adaptive selection | Unseen question, topic-coverage balancing, max Fisher information |
| Learning | Recompute ability with new evidence; optional held-out item calibration |
| API | FastAPI `/docs`, authenticated read/submit routes and admin routes |
| MySQL | SQLAlchemy Core tables; MySQL 8.4 in Compose |
| MongoDB | PyMongo collections with unique and query indexes; MongoDB 7 in Compose |
| Demo data | 60 reviewed-style sample MCQs; five simulated student profiles |
| Tests | Unit and API/SQL integration tests; real DB test optional |

## 2. Project layout

```text
app/
  api.py             FastAPI routes and API-key access control
  calibration.py     Optional, conservative item parameter recalibration
  config.py          Environment-backed database selection
  irt.py             Core 2PL probability, information, MAP theta, uncertainty
  mongo_store.py     MongoDB repository (PyMongo)
  mysql_store.py     MySQL repository (SQLAlchemy)
  schemas.py         Pydantic input validation and subject/topic taxonomy
  seed.py            60 MCQs, 5 synthetic learner profiles
  services.py        Reports, question selection and snapshot persistence
requirements.txt
requirements-dev.txt
.env.example
compose.yml
Dockerfile
tests/
docs/WHITEPAPER.md
```

## 3. Install on Linux (Python 3.11+ recommended)

From this directory:

```bash
cp .env.example .env
# Edit .env: replace BOTH example API keys with distinct long random secrets.
# For local development only, change the DB_BACKEND to mysql or mongodb.

docker compose up -d mysql mongodb
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**MySQL engine:** keep `DB_BACKEND=mysql` in `.env`.

```bash
python -m app.seed --demo
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

**MongoDB engine:** edit `.env` to `DB_BACKEND=mongodb`. MongoDB uses a **separate** database with its own seeded items and students.

```bash
python -m app.seed --demo
uvicorn app.api:app --host 127.0.0.1 --port 8001
```

API docs: `http://127.0.0.1:8000/docs` (MySQL) or `http://127.0.0.1:8001/docs` (MongoDB).

To launch both preconfigured API containers concurrently rather than local Uvicorn, after setting secrets in `.env`:

```bash
docker compose --profile api-mysql --profile api-mongo up -d --build
# MySQL API -> http://127.0.0.1:8000
# Mongo API -> http://127.0.0.1:8001

# Seed the two separate stores after first startup:
docker compose exec api-mysql python -m app.seed --demo
docker compose exec api-mongo python -m app.seed --demo
```

**Do not expose these development containers directly to the Internet.** The default MongoDB demonstration database has no authentication, and the API keys are service-level credentials rather than user/role authentication. Before production, use a locked-down network, DB credentials, TLS, secret storage, least privilege, per-student/teacher authorization, consent/retention controls, and rate limits.

## 4. API walk-through (real requests)

Set the keys by reading your edited `.env` file, or manually supply them:

```bash
export API_KEY='your-custom-api-key'
export ADMIN_API_KEY='your-custom-admin-key'
export BASE_URL=http://127.0.0.1:8000
```

**Create student** (the real learner's identifier is optional; use pseudonymous mapping):

```bash
curl -s -X POST "$BASE_URL/v1/students" \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"external_ref":"pupil-101","grade_level":10}'
# Save returned `student_id`
```

**Create item** (admin only):

```bash
curl -s -X POST "$BASE_URL/v1/admin/items" \
  -H "X-Admin-Key: $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"subject":"Maths","topic":"Algebra","question_text":"Solve 3x=12.",
       "choices":["2","3","4","6"],"correct_index":2,
       "difficulty":0.0,"discrimination":1.0}'
```

**Get next question** (only subject/topic, question and choices exposed):

```bash
curl -G "$BASE_URL/v1/next-item" \
  -H "X-API-Key: $API_KEY" \
  --data-urlencode 'student_id=PASTE_STUDENT_UUID' \
  --data-urlencode 'subject=Maths'
```

**Submit answer** (correctness is evaluated on the server):

```bash
curl -s -X POST "$BASE_URL/v1/responses" \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"student_id":"PASTE_STUDENT_UUID", "item_id":"PASTE_ITEM_UUID", "selected_index":2}'
```

**Read full report:**

```bash
curl "$BASE_URL/v1/students/PASTE_STUDENT_UUID/report" -H "X-API-Key: $API_KEY"
```

**Run optional offline calibration** after accumulating enough independent human observations:

```bash
python -m app.calibration --subject Maths
# Or authorized HTTP:
curl -X POST "$BASE_URL/v1/admin/calibrate/Maths" -H "X-Admin-Key: $ADMIN_API_KEY"
```

Automating calibration using cron is possible, but start with a reviewed, approval-gated batch process. Held-out loss improvement is necessary but not enough to certify educational validity.

## 5. Database schema

### MySQL tables

```text
tbl_students
  student_id [PK], external_ref, grade_level, created_at

tbl_assessment_items
  item_id [PK], subject, topic, question_text, choices [JSON], correct_index,
  difficulty, discrimination, initial_difficulty, initial_discrimination,
  parameter_version, calibration_n, calibrated_at, created_at

tbl_student_responses
  response_id [PK], student_id [FK], item_id [FK], selected_index,
  is_correct, answered_at, UNIQUE(student_id,item_id)

tbl_student_ability_scores
  student_id, subject, topic (empty => subject-level), theta,
  standard_error, ability_index_pct, response_count, evidence_status,
  model_version, updated_at, PRIMARY KEY(student_id,subject,topic)
```

SQLAlchemy creates these tables on startup. **Use Alembic/versioned migrations in production** for schema evolution.

### MongoDB collections

- `students`: `student_id`, optional `external_ref`, `grade_level`, `created_at`.
- `items`: question, choices, secure answer index, topic/subject, item parameters and metadata.
- `responses`: answer, correctness, time, and student/item IDs. Compound `(student_id,item_id)` unique index.
- `scores`: latest persisted subject/topic estimates, unique `(student_id,subject,topic)`.

The persisted scores are **materialized snapshots**. Authoritative truth is the response history plus current item parameters. Reports recompute and refresh their snapshots on access, so snapshots may be temporarily stale after recalibration until those students are assessed or queried again.

## 6. Model interpretation and responsible use

- `theta` is latent ability on an arbitrary IRT scale anchored here with a `N(0,1)` prior and provisional question parameters; it is **not** an intelligence quotient or a school grade.
- `ability_index_pct = 100 / (1+exp(-theta))` describes a **hypothetical reference item's** estimated correctness probability. It is **not a percentile, exam percent or an objective measure of general ability.**
- `ci95_theta` and `ci95_index_pct` are **Laplace approximate** posterior intervals; calibration and test-model uncertainty are not included.
- `observed_accuracy_pct` is raw accuracy on actually attempted questions and is a different statistic.
- `unassessed` returns null ability. `limited_evidence` / `exploratory_coverage` are **heuristic labels** about item count/content coverage, not certified reliability.
- The v1 model **uses individual question responses**. It does not infer IRT ability directly from historical subject averages or overall exam marks; incorporating such data requires test-form linkage or a validated grade observation model.
- Questions are multiple-choice only. Essay, pronunciation, oral English, lab skills and experimental reasoning require separate validated rubrics and scoring methods.
- Comparisons across grade levels and curricula require common-item anchoring, standard setting and equating, which are not implemented.
- If items leak or are repeatedly practiced, ability inference may become invalid. One answer per student/item is enforced for that reason; stronger item exposure and proctoring controls are needed for consequential use.

## 7. Test it

```bash
pip install -r requirements-dev.txt
pytest -q
```

SQLite is used as a temporary SQL compatibility database for the automated API tests; production targets MySQL. `tests/test_mongo.py` is optional and runs if `pymongo` and `mongomock` are installed. A live MySQL/MongoDB test with Compose is still recommended before deployment.

## 8. Extend the engine

To add curriculum topics: change `SUBJECT_TOPICS` in `app/schemas.py`, then add curated items in `app/seed.py` or through the admin API. For more complex assessment, extend item storage with curriculum identifiers, standard codes, grade, language, distractor statistics, cognitive dimension and accessibility accommodations.

For your school platform, map an existing student identifier to `external_ref` (or maintain a server-side ID mapping). Do not let a browser client access raw DB tables or read item answer keys. Teachers' dashboards can call the REST endpoints through your school's authenticated backend.

Refer to `docs/WHITEPAPER.md` for the equations, evaluation plan, model limitations and operational rollout.
