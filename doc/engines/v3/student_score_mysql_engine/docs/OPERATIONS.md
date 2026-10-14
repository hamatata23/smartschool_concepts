# Operating contract and limitations

## Forecast interpretation

- This is a numeric supervised regression baseline, not a language model or
  autonomous discovery system. It does not invent features, recover missing
  historical snapshots or guarantee improved accuracy on every retraining.
- The source may contain many fields. Only configured numeric inputs, IDs and
  relevant timestamps are read. Names/contact information are unnecessary.
- Numeric feature values must be finite, with identical definitions and units
  across historical and future records. NULL features are median-imputed using
  training rows. Malformed rows abort the run rather than silently biasing the data.
- The feature snapshot date must precede the target exam date. The engine cannot
  detect a false timestamp or every semantically leaked feature.
- Validation groups entire exam dates and permits the same student in earlier and
  later exams; this evaluates forecasting future exams, including known students.
  It does not estimate performance on wholly unseen schools/students. For that,
  add group-disjoint evaluation before relying on the model.
- A model is validated using an earlier-data fit, then refit on all labeled data.
  The saved production model includes validation labels. The report is an estimate
  of the training recipe's performance, not an independent evaluation of the final
  refit. Do not report training error as future prediction accuracy.
- Reusing validation cohorts for repeated decisions can bias evaluation. Keep a
  separate, untouched future cohort for periodic audits and monitor predictions
  against real scores. There is no automatic incumbent-versus-challenger accuracy
  comparison: an accepted model passes fixed gates and a mean baseline; it can
  still perform worse than the older model on future students.
- Coefficients are associations, not causal effects. Raising a value in the input
  does not prove an intervention will raise a student's real score. Use forecasts
  to support review and assistance, rather than automatically assign final grades.
- No calibrated confidence intervals or fairness guarantees are supplied.

## Transactions, identifiers and state

- Source is read-only. The output has one row per `(pipeline_name, source_row_id)`.
  Distinct exams must have distinct stable source IDs. IDs cannot be NULL, blank,
  padded with whitespace or longer than 128 characters.
- `pipeline_name` identifies a cohort/task, not a model release. Keep it stable
  across retraining. Use unique pipeline names and state directories per database
  and cohort. Run one scheduler host with persistent local model storage.
- Each run upserts current forecasts without duplicate rows. It marks earlier
  outputs `is_current=0` when their source is completed, removed, filtered out or
  the exam is on/past the current UTC date. It retains their last predicted values.
  It does not retain every historical revision of a forecast.
- `is_current` means current at the last successful run. Always inspect
  `predicted_at`, especially if your scheduler is stopped or failing. Failed runs
  leave the prior committed output unchanged; they do not magically invalidate it.
- During final prediction, the engine takes shared row locks on matching pending
  source records, computes predictions, marks old output inactive, and upserts the
  new output in one transaction. Teachers' conflicting edits wait until commit.
  Proper cohort indexes keep this transaction short. A lost DB connection fails
  the run; it never silently reconnects after losing its advisory lock.
- MySQL `GET_LOCK` prevents overlapping copies of the same pipeline on one MySQL
  server. A local file lock protects its state directory. This is not a distributed
  deployment design for NDB clusters or separately stored state on many workers.
- Files are JSON, not executable pickle. Writes use temporary files, fsync and
  atomic rename. An immutable `models/<version>.json` is written before the output
  table can reference that version. `state.json` contains the active model and
  latest attempted training fingerprint. `last_run.json` records run outcomes.
- Filesystem and MySQL do not form a single distributed transaction. If output
  commit fails after model persistence, the next run reuses that model and retries
  output. If commit succeeds but the process dies before its run report, rerunning
  the same pipeline is safe because output keys are unique.
- A corrected/deleted source row triggers a new training attempt. If that attempt
  fails the gates, the old model remains and may still reflect corrected/deleted
  data. If a deletion requires immediate removal of that influence, stop the
  service, archive its old model state according to your retention policy, and
  require a fresh accepted model before resuming.
- Changing features, source mapping, scope or score bounds makes the old model
  incompatible. It will not be used if a replacement fails. Training-only setting
  changes can fall back to the previous compatible model if rejected.

## Performance and scheduling

This is a bounded, in-memory batch engine: default at most 200,000 source rows per
cohort. It reads and hashes all matching rows each run to reliably detect updates
and deletions. It does not implement incremental SGD or change-data capture.
Memory, training cost and write duration increase with rows/features. Start with
a small representative cohort and measure before increasing `max_rows`.

For millions of rows, use a reviewed extraction/snapshot pipeline, partitioned
models, an evaluation store and incremental change capture. Do not remove the row
limit without capacity checks. Accepted model versions accumulate locally; include
them in backups and define retention after checking which versions outputs reference.

The default schedule is hourly. To change it, edit `OnCalendar` in the timer, then
run `systemctl daemon-reload` and restart the timer. Watch service failures through
your normal server monitoring. Exit 0 means predictions were written (possibly
using a retained old model); read the JSON training status for rejected candidates.
Exit 1 means the run failed or there is no usable model.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing columns or incompatible prediction schema | Compare `DESCRIBE` / `SHOW CREATE TABLE` output with the SQL and config. Have the DBA review any migration. |
| No accepted model | Read `state_dir/last_run.json`. Collect more dated actual results or improve inputs; review MAE settings without simply disabling evaluation. |
| Insufficient temporal data | Need enough earlier labels available before later feature snapshots. Two dates alone do not guarantee enough usable rows. |
| No predictions written | Need future `exam_date`, NULL `final_score`, NULL `score_recorded_at`, valid features and matching subject/grade filters. |
| TLS configuration error | Set MYSQL_SSL_CA to a trusted CA file and use a hostname that matches the server certificate. |
| Access denied / driver code | Check credentials and required grants with the DBA. Credentials and row values are not deliberately logged. |
| Another process is running | Wait for the active run; inspect service logs. Do not delete lock files to bypass a running process. |
| More features caused rejection | Populate historical inputs, inspect missing data and compare validation errors. More inputs are not always better. |
| Predictions seem stale | Check `predicted_at`, timer status, `last_run.json` and whether the latest training attempt was rejected. |
| Clock/time problems | Store timestamps in UTC. Source DATE columns have no timezone or exam time; same-day forecasts are deliberately excluded. |

Do not paste database passwords or identifiable student records into support
requests. Table definitions and anonymized sample rows are sufficient for mapping.

## References checked during implementation

- PyMySQL connection options, transactions and verified TLS:
  https://pymysql.readthedocs.io/en/latest/modules/connections.html
- MySQL connection-scoped advisory locks:
  https://dev.mysql.com/doc/refman/8.4/en/locking-functions.html
- MySQL upsert semantics and uniqueness:
  https://dev.mysql.com/doc/refman/8.4/en/insert-on-duplicate.html

The implementation uses bound values and repeated parameters in the update clause,
avoiding the deprecated `VALUES(column)` update form. SQL identifiers are restricted
to simple names; cross-schema dotted names are not accepted.
