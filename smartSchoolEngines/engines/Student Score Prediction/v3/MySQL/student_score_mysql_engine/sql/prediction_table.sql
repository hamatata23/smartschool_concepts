-- Required destination schema. No source changes.
CREATE TABLE IF NOT EXISTS `tbl_student_prediction_score` (
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
