-- EXAMPLE ONLY: run in a new/demo database. Adapt config.json for an existing table.
-- Store dates/times in UTC. One immutable pre-exam feature snapshot per student/exam.
CREATE TABLE IF NOT EXISTS tbl_student_score (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    student_id VARCHAR(128) NOT NULL,
    subject VARCHAR(64) NOT NULL,
    grade_level INT NOT NULL,
    exam_date DATE NOT NULL,
    features_as_of DATETIME NOT NULL,
    study_hours_per_week DOUBLE NULL,
    attendance_percent DOUBLE NULL,
    previous_score DOUBLE NULL,
    sleep_hours_per_night DOUBLE NULL,
    assignments_completed_percent DOUBLE NULL,
    final_score DOUBLE NULL,
    score_recorded_at DATETIME NULL,
    PRIMARY KEY (id),
    UNIQUE KEY student_exam (student_id, subject, grade_level, exam_date),
    KEY cohort_scores (subject, grade_level, final_score, exam_date),
    CHECK (final_score IS NULL OR final_score BETWEEN 0 AND 100),
    CHECK (features_as_of < exam_date),
    CHECK ((final_score IS NULL AND score_recorded_at IS NULL) OR
           (final_score IS NOT NULL AND score_recorded_at IS NOT NULL AND score_recorded_at >= exam_date))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
