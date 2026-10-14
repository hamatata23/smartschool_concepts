import copy
from contextlib import nullcontext
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import database
from demo_data import demo_rows
from engine import ScoreEngine
from learning import advance_state, contract, digest, normalize, predict_rows, temporal_split, train_candidate
from mysql_engine import atomic_json, local_lock, run_once
from settings import identifier, load_config, load_env, validate

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 14, 12)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "config.example.json")
        self.raw = demo_rows(NOW)
        self.rows, self.pending, _ = normalize(self.raw, self.c, NOW)

    def test_temporal_split_never_uses_later_labels(self):
        train, val, info = temporal_split(self.rows, self.c)
        self.assertEqual((len(train), len(val)), (320, 80))
        self.assertLess(max(r["target_date"] for r in train), min(r["target_date"] for r in val))
        self.assertTrue(all(r["label_available_at"] <= info["training_label_cutoff_utc"] for r in train))

    def test_late_recorded_results_are_purged(self):
        _, val, _ = temporal_split(self.rows, self.c)
        self.rows[0]["label_available_at"] = val[0]["target_date"] + "T00:00:00"
        train, _, report = temporal_split(self.rows, self.c)
        self.assertEqual(len(train), 319)
        self.assertEqual(report["purged_rows"], 1)

    def test_preprocessing_uses_training_subset(self):
        train, val, _ = temporal_split(self.rows, self.c)
        for r in val:
            r["study_hours_per_week"] = 150
        c = copy.deepcopy(self.c)
        c["training"]["max_validation_mae"] = 100
        c["training"]["require_better_than_mean_baseline"] = False
        model, report = train_candidate(train + val, c, NOW)
        self.assertEqual(report["candidate_training"]["rows"], 320)
        # Independent train-only fit must reproduce the measured holdout result.
        from learning import fit_model
        independent = fit_model(train, c).evaluate(val, "actual_score")
        self.assertEqual(report["validation"], independent)
        self.assertEqual(model["model"]["training"]["rows"], 400)

    def test_good_model_refits_all_labels_and_predicts(self):
        state, report = advance_state(None, self.rows, self.c, now=NOW)
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(state["active"]["training_rows"], 400)
        self.assertLess(state["active"]["validation_mae"], 3)
        result = predict_rows(self.pending, state["active"], self.c)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0]["predicted_score"], 67.90883, places=4)

    def test_unchanged_labels_do_not_retrain(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        again, report = advance_state(state, self.rows, self.c, now=NOW)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(state["active"]["version"], again["active"]["version"])

    def test_actual_correction_triggers_retraining(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        self.rows[0]["actual_score"] += 1
        again, report = advance_state(state, self.rows, self.c, now=NOW)
        self.assertEqual(report["status"], "accepted")
        self.assertNotEqual(state["active"]["version"], again["active"]["version"])

    def test_new_label_and_deletion_change_fingerprint(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        fewer, _ = advance_state(state, self.rows[1:], self.c, now=NOW)
        self.assertNotEqual(state["last_attempt"]["fingerprint"], fewer["last_attempt"]["fingerprint"])
        new = dict(self.rows[0], row_id="new", student_id="new")
        more, _ = advance_state(state, self.rows+[new], self.c, now=NOW)
        self.assertEqual(more["active"]["training_rows"], 401)

    def test_pending_updates_only_change_predictions(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        a = predict_rows(self.pending, state["active"], self.c)
        self.pending[0]["previous_score"] += 10
        b = predict_rows(self.pending, state["active"], self.c)
        self.assertGreater(b[0]["predicted_score"], a[0]["predicted_score"])
        self.assertNotEqual(b[0]["input_hash"], a[0]["input_hash"])
        self.assertEqual(advance_state(state, self.rows, self.c)[1]["status"], "unchanged")

    def test_rejected_model_preserves_previous_version(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        self.c["training"]["max_validation_mae"] = 0
        again, report = advance_state(state, self.rows, self.c, now=NOW)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(again["active"]["version"], state["active"]["version"])
        self.assertEqual(advance_state(again, self.rows, self.c)[1]["status"], "unchanged")

    def test_feature_change_cannot_reuse_incompatible_model(self):
        state, _ = advance_state(None, self.rows, self.c, now=NOW)
        self.c["features"].append("quiz_average")
        for r in self.rows:
            r["quiz_average"] = None
        again, report = advance_state(state, self.rows, self.c, now=NOW)
        self.assertIsNone(again["active"])
        self.assertEqual(report["status"], "rejected")

    def test_insufficient_history_and_single_date(self):
        for rows in ([], self.rows[:10], [dict(r, target_date="2025-01-01") for r in self.rows]):
            with self.subTest(count=len(rows)):
                state, report = advance_state(None, rows, self.c, now=NOW)
                self.assertIsNone(state["active"])
                self.assertEqual(report["status"], "rejected")

    def test_null_is_unknown_but_zero_is_real(self):
        raw = copy.deepcopy(self.raw)
        raw[0]["final_score"] = 0
        labeled, pending, _ = normalize(raw, self.c, NOW)
        self.assertEqual(len(labeled), 400)
        self.assertEqual(len(pending), 5)
        self.assertEqual(next(r for r in labeled if r["row_id"] == "1")["actual_score"], 0)

    def test_invalid_rows_fail_closed(self):
        cases = [("features_as_of", NOW+timedelta(days=1)), ("final_score", 101),
                 ("score_recorded_at", None), ("sleep_hours_per_night", 25),
                 ("previous_score", float("inf")), ("student_id", " x ")]
        for field, value in cases:
            raw = copy.deepcopy(self.raw)
            raw[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize(raw, self.c, NOW)
        with self.assertRaises(ValueError):
            normalize(self.raw+[self.raw[0]], self.c, NOW)

    def test_missing_feature_imputed_but_all_missing_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw[-1]["sleep_hours_per_night"] = None
        rows, pending, _ = normalize(raw, self.c, NOW)
        state, _ = advance_state(None, rows, self.c, now=NOW)
        details = predict_rows(pending, state["active"], self.c)
        self.assertEqual(details[-1]["imputed_features"], ["sleep_hours_per_night"])
        for f in self.c["features"]:
            raw[-1][f] = None
        with self.assertRaises(ValueError):
            normalize(raw, self.c, NOW)

    def test_same_day_and_past_unknown_exams_not_predicted(self):
        raw = copy.deepcopy(self.raw)
        raw[-1]["exam_date"] = NOW.date()
        raw[-1]["features_as_of"] = NOW-timedelta(days=1)
        _, pending, skipped = normalize(raw, self.c, NOW)
        self.assertEqual((len(pending), skipped), (4, 1))


class IntegrationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "config.example.json")

    def test_sql_identifiers_and_filter_binding(self):
        with self.assertRaises(ValueError): identifier("x; DROP TABLE students")
        c = copy.deepcopy(self.c)
        value = "mathematics' OR 1=1 --"
        c["filter_equals"]["subject"] = value
        sql, args = database.source_query(c)
        self.assertNotIn(value, sql)
        self.assertIn(value, args)
        self.assertNotIn(c["prediction_table"], sql)
        self.assertNotIn("SELECT *", sql)

    def test_output_upsert_has_keys_and_bound_values(self):
        rows, pending, _ = normalize(demo_rows(NOW), self.c, NOW)
        state, _ = advance_state(None, rows, self.c, now=NOW)
        active = state["active"]
        results = predict_rows(pending, active, self.c)
        conn = MagicMock()
        database.write_predictions(conn, self.c, results, active, NOW)
        calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
        self.assertEqual(len(calls), 6)
        sql, values = calls[1].args
        self.assertIn("ON DUPLICATE KEY UPDATE", sql)
        self.assertEqual(sql.count("%s"), len(values))
        self.assertEqual(values[1], "401")
        self.assertEqual(values[6], active["version"])
        conn.commit.assert_not_called()

    def test_atomic_model_state_round_trip(self):
        rows, _, _ = normalize(demo_rows(NOW), self.c, NOW)
        state, _ = advance_state(None, rows, self.c, now=NOW)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            atomic_json(p, state)
            loaded = json.loads(p.read_text())
            model = ScoreEngine.from_dict(loaded["active"]["model"])
            self.assertEqual(model.training["rows"], 400)
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_local_lock_refuses_overlap(self):
        with tempfile.TemporaryDirectory() as directory, local_lock(directory):
            with self.assertRaises(RuntimeError), local_lock(directory):
                pass

    def test_config_rejects_target_features_and_output_as_source(self):
        for mutate in (lambda c: c["features"].append("final_score"),
                       lambda c: c.update(prediction_table=c["source_table"]),
                       lambda c: c["features"].append("actual_score")):
            c = copy.deepcopy(self.c)
            mutate(c)
            with self.assertRaises(ValueError): validate(c)

    def test_env_file_is_literal_and_preserves_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TEST_EXISTING": "keep"}):
            p = Path(directory)/"env"
            p.write_text("TEST_EXISTING=replace\nTEST_PASSWORD=$(whoami)#literal\n")
            load_env(p)
            self.assertEqual(os.environ["TEST_EXISTING"], "keep")
            self.assertEqual(os.environ["TEST_PASSWORD"], "$(whoami)#literal")

    def test_workflow_commits_and_reuses_persisted_model(self):
        raw = demo_rows(NOW)
        with tempfile.TemporaryDirectory() as directory:
            c = dict(self.c, state_dir=directory)
            conn = MagicMock()
            with patch("database.database_lock", return_value=nullcontext()), patch("database.check_schema"), \
                 patch("database.fetch_source", side_effect=[raw, raw[-5:], raw, raw[-5:]]), \
                 patch("database.write_predictions") as write:
                first = run_once(conn, c, now=NOW)
                second = run_once(conn, c, now=NOW)
            self.assertEqual(first["predictions_written"], 5)
            self.assertEqual(first["active_model_version"], second["active_model_version"])
            self.assertEqual(second["training"]["status"], "unchanged")
            self.assertEqual(write.call_count, 2)
            self.assertEqual(len(list((Path(directory)/"models").glob("*.json"))), 1)

    def test_database_failure_rolls_back_but_keeps_recoverable_model(self):
        raw = demo_rows(NOW)
        with tempfile.TemporaryDirectory() as directory:
            c = dict(self.c, state_dir=directory)
            conn = MagicMock()
            with patch("database.database_lock", return_value=nullcontext()), patch("database.check_schema"), \
                 patch("database.fetch_source", side_effect=[raw, raw[-5:]]), \
                 patch("database.write_predictions", side_effect=RuntimeError("temporary DB error")):
                with self.assertRaises(RuntimeError): run_once(conn, c, now=NOW)
            conn.rollback.assert_called_once()
            # Only the initial read commits; no partial output commit.
            conn.commit.assert_called_once()
            self.assertTrue((Path(directory)/"state.json").exists())
            report = json.loads((Path(directory)/"last_run.json").read_text())
            self.assertEqual(report["status"], "prediction_transaction_failed")


if __name__ == "__main__":
    unittest.main()
