# Monitoring & Verification Screenshots

This directory stores the visual verification artifacts required by Stage G of the assignment.

### Required Screenshots Checklist

1. **`01_dag_graph_data_pipeline.png`**
   - **View**: Graph View of `data_pipeline`
   - **URL**: [http://localhost:8080/dags/data_pipeline/graph](http://localhost:8080/dags/data_pipeline/graph)
   - **Content**: Shows the 4 tasks (`extract_to_bronze` -> `transform_to_silver` -> `prepare_gold` -> `validate_gold`) with dependencies and successful green execution borders.

2. **`02_dag_graph_ml_pipeline.png`**
   - **View**: Graph View of `ml_training_pipeline`
   - **URL**: [http://localhost:8080/dags/ml_training_pipeline/graph](http://localhost:8080/dags/ml_training_pipeline/graph)
   - **Content**: Shows the 4 ML tasks (`load_gold_data` -> `train_model` -> `evaluate_model` -> `save_model`) in successful green state.

3. **`03_three_date_backfill_grid.png`**
   - **View**: Grid View of `data_pipeline`
   - **URL**: [http://localhost:8080/dags/data_pipeline/grid](http://localhost:8080/dags/data_pipeline/grid)
   - **Content**: Shows the runs for the three backfilled dates (`2026-10-18`, `2026-10-19`, `2026-10-20`) all passing with green checkmarks.

4. **`04_retry_evidence.png`**
   - **View**: Task Instance Details / Log for `extract_to_bronze` on `2026-10-21`
   - **Content**: Shows `attempt 1` failing with `RuntimeError: [Simulated Retry Test] Transient network/storage glitch...` and transitioning to `UP_FOR_RETRY`, followed by `attempt 2` succeeding.

5. **`05_same_date_idempotent_rerun.png`**
   - **View**: Terminal CLI execution or Airflow task log
   - **Content**: Confirms re-running date `2026-10-18` outputs exactly 245 rows without duplicates.

6. **`06_ml_pipeline_execution_logs.png`**
   - **View**: Task instance logs for `evaluate_model` and `save_model`
   - **Content**: Demonstrates model evaluation metrics (Accuracy, Precision, Recall, F1) and serialization to `models/delivery_delay_model.joblib`.
