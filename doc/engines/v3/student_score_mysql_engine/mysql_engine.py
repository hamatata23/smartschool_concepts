"""Unattended MySQL score prediction: run once from systemd, or manually."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

import database
from learning import advance_state, normalize, predict_rows, utcnow
from settings import load_config, load_env


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(prefix=".save-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def local_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(directory / "run.lock", "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this state directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_once(conn, c, force=False, now=None):
    directory = Path(c["state_dir"])
    fixed_now = now
    with local_lock(directory), database.database_lock(conn, c):
        database.check_schema(conn, c)
        # One SELECT snapshot, not a long transaction while numerical fitting runs.
        raw = database.fetch_source(conn, c)
        conn.commit()
        now = fixed_now or utcnow()
        labeled, _, skipped = normalize(raw, c, now)
        path = directory / "state.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        state, training_report = advance_state(previous, labeled, c, force, now)
        active = state.get("active")
        if active:
            version_path = directory / "models" / (active["version"] + ".json")
            if not version_path.exists():
                atomic_json(version_path, active)
        # Persist the immutable model before any DB result can reference its ID.
        atomic_json(path, state)
        result = {"run_at_utc": now.isoformat(), "labeled_rows": len(labeled),
                  "skipped_past_unlabeled_rows": skipped, "training": training_report,
                  "active_model_version": active["version"] if active else None}
        if not active:
            atomic_json(directory / "last_run.json", {**result, "status": "no_accepted_model"})
            raise RuntimeError("No accepted model. Check state_dir/last_run.json for training requirements or rejected validation")
        # Re-read and lock pending source records briefly so teachers cannot edit
        # their features/results between prediction and output commit.
        try:
            current = database.fetch_source(conn, c, pending_only=True, locked=True)
            prediction_now = fixed_now or utcnow()
            _, pending, _ = normalize(current, c, prediction_now)
            predictions = predict_rows(pending, active, c)
            database.write_predictions(conn, c, predictions, active, prediction_now)
            conn.commit()
        except Exception:
            conn.rollback()
            atomic_json(directory / "last_run.json", {**result, "status": "prediction_transaction_failed"})
            raise
        result.update(status="ok", predictions_written=len(predictions))
        atomic_json(directory / "last_run.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "init-output", "run-once"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--env-file", help="Literal KEY=value file; optional if environment is already set")
    parser.add_argument("--force-retrain", action="store_true", help="Retry training even when data has not changed")
    args = parser.parse_args(argv)
    conn = None
    try:
        load_env(args.env_file)
        c = load_config(args.config)
        conn = database.connect()
        if args.command == "init-output":
            database.init_output(conn, c)
            database.check_schema(conn, c)
            result = {"status": "ok", "prediction_table": c["prediction_table"]}
        elif args.command == "check":
            database.check_schema(conn, c)
            raw = database.fetch_source(conn, c)
            labeled, pending, skipped = normalize(raw, c)
            conn.rollback()
            result = {"status": "ok", "labeled_rows": len(labeled), "future_unlabeled_rows": len(pending),
                      "past_unlabeled_rows": skipped}
        else:
            result = run_once(conn, c, args.force_retrain)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Driver messages can include data. Log only exception type and numeric code.
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
        print(f"ERROR: {type(exc).__name__}, code={code}; check DB access, schema, and server logs", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
