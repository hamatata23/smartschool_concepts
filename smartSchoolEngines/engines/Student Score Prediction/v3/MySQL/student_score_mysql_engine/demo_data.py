"""Synthetic records for tests/demo only; never use them to evaluate real accuracy."""
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import numpy as np


def demo_rows(now=None):
    now = (now or datetime.utcnow()).replace(hour=0, minute=0, second=0, microsecond=0)
    rng = np.random.default_rng(74)
    rows = []
    for year in range(5):
        exam = now - timedelta(days=365*(5-year))
        for student in range(80):
            h, a, p, s, hw = map(float, rng.uniform([1, 45, 20, 4, 30], [30, 100, 95, 9, 100]))
            score = float(np.clip(3 + .40*h + .14*a + .49*p + 1.2*s + .10*hw + rng.normal(0, 2), 0, 100))
            rows.append(dict(id=len(rows)+1, student_id=f"DEMO-{year}-{student}", subject="mathematics",
                grade_level=8, exam_date=exam.date(), features_as_of=exam-timedelta(days=60),
                study_hours_per_week=h, attendance_percent=a, previous_score=p,
                sleep_hours_per_night=s, assignments_completed_percent=hw,
                final_score=score, score_recorded_at=exam+timedelta(days=1)))
    for student in range(5):
        rows.append(dict(id=len(rows)+1, student_id=f"DEMO-FUTURE-{student}", subject="mathematics",
            grade_level=8, exam_date=(now+timedelta(days=60)).date(), features_as_of=now,
            study_hours_per_week=12+student, attendance_percent=85+student,
            previous_score=65+student, sleep_hours_per_night=7,
            assignments_completed_percent=80+student, final_score=None, score_recorded_at=None))
    return rows


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def write_sql(path):
    rows = demo_rows()
    # Omit IDs so that the demo database assigns them. A rerun fails on the
    # student_exam unique key rather than quietly mixing duplicate training data.
    names = [n for n in rows[0] if n != "id"]
    sql = "-- SYNTHETIC DEMO ONLY. Run once in a disposable database.\nSTART TRANSACTION;\n"
    for row in rows:
        sql += "INSERT INTO tbl_student_score (" + ",".join(names) + ") VALUES (" + ",".join(sql_literal(row[n]) for n in names) + ");\n"
    sql += "COMMIT;\n"
    Path(path).write_text(sql, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo_students.sql")
    args = parser.parse_args()
    write_sql(args.output)
    print(f"Created {args.output}: 400 synthetic historical rows and 5 upcoming records")
