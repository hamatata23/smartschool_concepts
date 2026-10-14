"""MySQL 8.0.22+ adapter. Values are bound parameters; identifiers are validated."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os

from settings import identifier as q


OUTPUT_COLUMNS = {
    "pipeline_name", "source_row_id", "student_id", "target_date", "predicted_score", "raw_score",
    "model_version", "input_hash", "training_rows", "validation_mae", "warnings_json", "is_current", "predicted_at"
}


def connect():
    import pymysql
    required = ("MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
    if any(not os.environ.get(k) for k in required):
        raise ValueError("Set MYSQL_DATABASE, MYSQL_USER, and MYSQL_PASSWORD")
    options = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")), user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"], database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor, autocommit=False,
        connect_timeout=10, read_timeout=120, write_timeout=120, local_infile=False)
    if os.environ.get("MYSQL_UNIX_SOCKET"):
        options["unix_socket"] = os.environ["MYSQL_UNIX_SOCKET"]
    if os.environ.get("MYSQL_SSL_CA"):
        options.update(ssl_ca=os.environ["MYSQL_SSL_CA"], ssl_verify_cert=True, ssl_verify_identity=True)
    elif options["host"] not in ("localhost", "127.0.0.1", "::1") and not options.get("unix_socket"):
        raise ValueError("Set MYSQL_SSL_CA for a remote MySQL server")
    conn = pymysql.connect(**options)
    try:
        with conn.cursor() as cur:
            cur.execute("SET time_zone = '+00:00'")
            cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cur.execute("SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'")
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise


def source_query(c, pending_only=False, locked=False):
    names = list(dict.fromkeys([*c["columns"].values(), *c["features"]]))
    clauses, args = [], []
    for name, value in c["filter_equals"].items():
        if value is None:
            clauses.append(f"{q(name)} IS NULL")
        else:
            clauses.append(f"{q(name)} = %s")
            args.append(value)
    if pending_only:
        clauses.append(f"{q(c['columns']['target'])} IS NULL")
    sql = f"SELECT {', '.join(q(n) for n in names)} FROM {q(c['source_table'])}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {q(c['columns']['row_id'])} LIMIT %s"
    args.append(c["max_rows"] + 1)
    if locked:
        sql += " FOR SHARE"
    return sql, tuple(args)


def fetch_source(conn, c, pending_only=False, locked=False):
    sql, args = source_query(c, pending_only, locked)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    if len(rows) > c["max_rows"]:
        raise ValueError("Source exceeds max_rows; partition by cohort or increase the reviewed memory limit")
    return rows


def output_ddl(c):
    return f"""CREATE TABLE IF NOT EXISTS {q(c['prediction_table'])} (
        pipeline_name VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
        source_row_id VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
        student_id VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
        target_date DATE NOT NULL,
        predicted_score DOUBLE NOT NULL,
        raw_score DOUBLE NOT NULL,
        model_version CHAR(36) CHARACTER SET ascii NOT NULL,
        input_hash CHAR(64) CHARACTER SET ascii NOT NULL,
        training_rows INT UNSIGNED NOT NULL,
        validation_mae DOUBLE NOT NULL,
        warnings_json JSON NOT NULL,
        is_current TINYINT(1) NOT NULL DEFAULT 1,
        predicted_at DATETIME(6) NOT NULL,
        PRIMARY KEY (pipeline_name, source_row_id),
        KEY lookup_student (student_id, target_date),
        KEY lookup_current (pipeline_name, is_current)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin"""


def init_output(conn, c):
    with conn.cursor() as cur:
        cur.execute(output_ddl(c))
    conn.commit()


def check_schema(conn, c):
    """No automatic ALTER or changes to source; reject incompatible existing output."""
    with conn.cursor() as cur:
        for table in (c["source_table"], c["prediction_table"]):
            cur.execute("SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (table,))
            row = cur.fetchone()
            if not row or row["ENGINE"] != "InnoDB":
                raise ValueError(f"{table} must exist and use InnoDB (run init-output for a new output table)")
        cur.execute(f"SHOW COLUMNS FROM {q(c['source_table'])}")
        source = {r["Field"]: r for r in cur.fetchall()}
        needed = {*c["columns"].values(), *c["features"], *c["filter_equals"]}
        if needed - source.keys():
            raise ValueError("Source table is missing columns: " + ", ".join(sorted(needed-source.keys())))
        if source[c["columns"]["target_date"]]["Type"].lower() != "date":
            raise ValueError("Configured target_date column must have MySQL DATE type")
        for key in ("features_as_of", "label_available_at"):
            typ = source[c["columns"][key]]["Type"].lower()
            if not typ.startswith(("datetime", "timestamp")):
                raise ValueError(f"{key} column must be DATETIME or TIMESTAMP")
        cur.execute(f"SHOW INDEX FROM {q(c['source_table'])}")
        unique = {}
        for r in cur.fetchall():
            if not r["Non_unique"]:
                unique.setdefault(r["Key_name"], []).append((r["Seq_in_index"], r["Column_name"], r["Sub_part"]))
        if not any(v == [(1, c["columns"]["row_id"], None)] for v in unique.values()):
            raise ValueError("Source row_id needs a full-column unique or primary index")
        cur.execute(f"SHOW FULL COLUMNS FROM {q(c['prediction_table'])}")
        output = {r["Field"]: r for r in cur.fetchall()}
        if OUTPUT_COLUMNS - output.keys():
            raise ValueError("Existing prediction table differs from required schema; review sql/prediction_table.sql")
        expected_types = {"pipeline_name": "varchar(64)", "source_row_id": "varchar(128)",
            "student_id": "varchar(128)", "target_date": "date", "predicted_score": "double",
            "raw_score": "double", "model_version": "char(36)", "input_hash": "char(64)",
            "training_rows": "int unsigned", "validation_mae": "double", "warnings_json": "json",
            "is_current": "tinyint(1)", "predicted_at": "datetime(6)"}
        for name, typ in expected_types.items():
            if output[name]["Type"].lower() != typ or output[name]["Null"] != "NO":
                raise ValueError(f"Prediction column {name} must be {typ} NOT NULL")
        for name, collation in (("pipeline_name", "ascii_bin"), ("source_row_id", "utf8mb4_bin")):
            if output[name]["Collation"] != collation:
                raise ValueError(f"Prediction key {name} must use {collation} collation")
        cur.execute(f"SHOW INDEX FROM {q(c['prediction_table'])}")
        indexes = cur.fetchall()
        primary = sorted((r["Seq_in_index"], r["Column_name"], r["Sub_part"]) for r in indexes if r["Key_name"] == "PRIMARY")
        if primary != [(1, "pipeline_name", None), (2, "source_row_id", None)]:
            raise ValueError("Prediction primary key must be (pipeline_name, source_row_id)")
        if any(not r["Non_unique"] and r["Key_name"] != "PRIMARY" for r in indexes):
            raise ValueError("Remove extra UNIQUE indexes from prediction table after review; they can merge separate forecasts")
    conn.commit()


@contextmanager
def database_lock(conn, c):
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE() AS name")
        database = cur.fetchone()["name"]
    name = "scores:" + hashlib.sha256((database + ":" + c["prediction_table"].lower() + ":" + c["pipeline_name"]).encode()).hexdigest()[:56]
    with conn.cursor() as cur:
        cur.execute("SELECT GET_LOCK(%s, 0) AS acquired", (name,))
        if cur.fetchone()["acquired"] != 1:
            raise RuntimeError("Another process is running this prediction pipeline")
    try:
        yield
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT RELEASE_LOCK(%s)", (name,))


def write_predictions(conn, c, predictions, active, now):
    """Caller owns commit/rollback. Exactly one current result per source record."""
    names = ["pipeline_name", "source_row_id", "student_id", "target_date", "predicted_score", "raw_score",
             "model_version", "input_hash", "training_rows", "validation_mae", "warnings_json", "is_current", "predicted_at"]
    update_names = names[2:]
    # Repeat bound parameters in the update clause: no deprecated VALUES(col).
    sql = (f"INSERT INTO {q(c['prediction_table'])} ({', '.join(q(n) for n in names)}) "
           f"VALUES ({', '.join(['%s']*len(names))}) ON DUPLICATE KEY UPDATE "
           + ", ".join(f"{q(n)} = %s" for n in update_names))
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {q(c['prediction_table'])} SET is_current=0 WHERE pipeline_name=%s AND is_current=1", (c["pipeline_name"],))
        for p in predictions:
            warnings = {k: p[k] for k in ("clipped", "imputed_features", "outside_training_range")}
            values = (c["pipeline_name"], p["source_row_id"], p["student_id"], p["target_date"],
                p["predicted_score"], p["raw_score"], active["version"], p["input_hash"],
                active["training_rows"], active["validation_mae"], json.dumps(warnings), 1, now)
            cur.execute(sql, values + values[2:])
