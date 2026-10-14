"""Opt-in real MySQL integration test. Requires a disposable *_engine_test database."""
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
import unittest
import uuid

import database
from demo_data import demo_rows
from mysql_engine import run_once
from settings import load_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("RUN_MYSQL_INTEGRATION") == "1", "Live MySQL is opt-in")
class LiveMySQLTests(unittest.TestCase):
    def test_train_predict_repeat_update_and_complete(self):
        self.assertTrue(os.environ.get("MYSQL_DATABASE", "").endswith("_engine_test"),
                        "Integration test requires a disposable database ending in _engine_test")
        c = load_config(ROOT/"config.example.json")
        suffix = uuid.uuid4().hex[:10]
        c.update(source_table="test_score_"+suffix, prediction_table="test_prediction_"+suffix)
        conn = database.connect()
        now = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
        try:
            with conn.cursor() as cur:
                ddl = (ROOT/"sql/example_source.sql").read_text().replace("tbl_student_score", c["source_table"])
                cur.execute(ddl)
                rows = demo_rows(now)
                names = list(rows[0])
                cur.executemany(f"INSERT INTO `{c['source_table']}` ({','.join('`'+x+'`' for x in names)}) VALUES ({','.join(['%s']*len(names))})",
                                [tuple(r[n] for n in names) for r in rows])
            conn.commit()
            database.init_output(conn, c)
            with tempfile.TemporaryDirectory() as directory:
                c["state_dir"] = directory
                first = run_once(conn, c, now=now)
                second = run_once(conn, c, now=now)
                self.assertEqual(first["predictions_written"], 5)
                self.assertEqual(first["active_model_version"], second["active_model_version"])
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) AS n FROM `{c['prediction_table']}`")
                    self.assertEqual(cur.fetchone()["n"], 5)
                    cur.execute(f"UPDATE `{c['source_table']}` SET previous_score=90 WHERE id=401")
                    cur.execute(f"UPDATE `{c['source_table']}` SET final_score=final_score+1 WHERE id=1")
                conn.commit()
                third = run_once(conn, c, now=now)
                self.assertNotEqual(third["active_model_version"], first["active_model_version"])
                with conn.cursor() as cur:
                    cur.execute(f"UPDATE `{c['source_table']}` SET final_score=70, score_recorded_at=%s WHERE id=401", (now+timedelta(days=61),))
                conn.commit()
                fourth = run_once(conn, c, now=now+timedelta(days=62))
                self.assertEqual(fourth["predictions_written"], 0)
                with conn.cursor() as cur:
                    cur.execute(f"SELECT SUM(is_current) AS n FROM `{c['prediction_table']}`")
                    self.assertEqual(cur.fetchone()["n"], 0)
        finally:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS `{c['prediction_table']}`")
                cur.execute(f"DROP TABLE IF EXISTS `{c['source_table']}`")
            conn.commit()
            conn.close()
