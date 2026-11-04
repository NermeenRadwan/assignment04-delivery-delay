# Assignment 04: From Daily CSVs to Automated ML Pipelines

An end-to-end, production-grade, date-aware Data Engineering and MLOps pipeline designed to predict late e-commerce deliveries (`late_delivery`: 0 = On Time, 1 = Late) using Apache Airflow, Docker Compose, Pandas, and Scikit-learn.

---

## 1. Project Objective & Business Scenario

### Business Scenario
In an e-commerce logistics network operating across Egypt (Cairo, Giza, Alexandria, Delta, Upper Egypt, Canal), timely delivery is critical for customer retention. Logistics operators receive daily batches of placed orders and need to:
1. Automatically ingest, clean, and validate daily operational batches.
2. Flag orders at high risk of delivery delays (`late_delivery = 1`) at the time of placement.
3. Allow fulfillment managers to proactively intervene (e.g., prioritize warehouse picking or allocate faster carriers).

### Primary Engineering Objectives
- **Medallion Architecture**: Implement Bronze (raw copy), Silver (deduplicated & sanitized), and Gold (feature-engineered) layers.
- **Airflow Orchestration**: Automate daily execution using date-aware processing (`{{ ds }}`) and ensure complete pipeline idempotency.
- **ML Lifecycle**: Build a leak-free Scikit-learn Pipeline with `ColumnTransformer` to train and evaluate a `RandomForestClassifier`.
- **Pipeline Reliability**: Demonstrate resilience via automated retries (`retries=2`), idempotency on rerun, and multi-date historical backfilling.

---

## 2. Dataset Description & Exploration Findings

The dataset consists of **46 daily CSV files** (`orders_2026-10-01.csv` to `orders_2026-11-15.csv`) totaling **10,219 records**.

### Columns & Metadata (15 original columns)
- **Identifiers & Temporal**:
  - `order_id`: Unique identifier (`ORD-YYYYMMDD-XXXX`). Contains duplicated records across batches (~3 per day).
  - `order_date`: Batch placement date (`YYYY-MM-DD`). Matches daily partition.
  - `order_hour`: Hour placed (0–23).
- **Categoricals**:
  - `customer_region`: Destination region (`Cairo`, `Giza`, `Alexandria`, `Delta`, `Upper_Egypt`, `Canal`).
  - `warehouse_id`: Fulfillment center (`WH-01`, `WH-02`, `WH-03`, `WH-04`).
  - `shipping_method`: Shipping service tier (`Standard`, `Express`, `Same_Day`).
  - `weather_risk_level`: Forecast weather disruption (`Low`, `Medium`, `High`, missing).
- **Numericals**:
  - `distance_km`: Trip distance in kilometers.
  - `items_count`: Quantity of items ordered.
  - `package_weight_kg`: Total parcel weight in kg.
  - `order_value_egp`: Total monetary value in Egyptian Pounds.
  - `promised_delivery_days`: Promised fulfillment window (1–5 days).
  - `warehouse_load_pct`: Fulfillment center utilization percentage (0–100%).
  - `carrier_on_time_rate`: Historical carrier reliability metric (0.0–1.0).
- **Target**:
  - `late_delivery`: Binary target (`0` = on time: 76.33%, `1` = late: 23.67%).

### Exploration & Quality Findings
1. **Duplicates**: 166 duplicate rows discovered across all files (~3 duplicate rows per day). The duplicates are exact identical rows sharing the same `order_id`.
2. **Corrupt Sentinel Values**:
   - `distance_km` contains corrupt negative values (`-8.0`).
   - `order_value_egp` contains corrupt negative values (`-110.0`).
   - `items_count` contains invalid zero values (`0`). An order must contain at least 1 item.
3. **Inconsistent Casing / Whitespace**:
   - `shipping_method` contains strings with whitespace (`" standard "`) and varying casing (`"same_day"`, `"EXPRESS"`).
4. **Missing Values**:
   - Missing entries observed in `weather_risk_level` (0.92%), `distance_km` (1.39%), `warehouse_load_pct` (1.46%), `carrier_on_time_rate` (0.85%), and `shipping_method` (0.72%).

---

## 3. Data Architecture: Medallion Transformations

The data pipeline in `src/etl.py` processes **one specific date at a time** using modular, reusable functions.

### Bronze Layer (`extract_to_bronze`)
- Copies the raw daily CSV (`data/raw/orders_{date}.csv`) into `data/bronze/orders_{date}.csv`.
- Preserves all duplicate records and corrupt values verbatim to maintain an immutable raw audit trail.
- Overwrites the target date file to guarantee idempotency.

### Silver Layer (`transform_to_silver`)
- **Deduplication**: Removes duplicate records using `order_id` (keeps first).
- **String Normalization**: Trims leading/trailing whitespace and standardizes casing for `shipping_method` (`Standard`, `Express`, `Same_Day`), `weather_risk_level` (`Low`, `Medium`, `High`, `Unknown`), `customer_region`, and `warehouse_id`.
- **Domain Cleaning**:
  - Invalid negative `distance_km < 0` $\rightarrow$ set to `NaN` (handled by imputer).
  - Invalid negative `order_value_egp < 0` $\rightarrow$ set to `NaN`.
  - Invalid `items_count <= 0` $\rightarrow$ clipped to `1`.
  - Invalid `promised_delivery_days < 1` $\rightarrow$ clipped to `1`.
  - Out-of-bounds `carrier_on_time_rate` outside `[0, 1]` $\rightarrow$ set to `NaN`.
- **Type Casting**: Explicit casting of integers (`order_hour`, `items_count`, `promised_delivery_days`, `late_delivery`), floats, and strings.

### Gold Layer (`prepare_gold`)
Prepares clean feature tables and engineers 3 domain features derived at order placement time without target leakage:
1. **`distance_per_promised_day`** = $\frac{\text{distance\_km}}{\text{promised\_delivery\_days}}$: Measures daily transit urgency. High km/day strongly correlates with delivery delays.
2. **`weight_per_item`** = $\frac{\text{package\_weight\_kg}}{\text{items\_count}}$: Measures package density and handling difficulty.
3. **`is_rush_order`** = $(\text{promised\_delivery\_days} == 1 \lor \text{shipping\_method} == \text{"Same\_Day"})$: Binary flag denoting urgent fulfillment requirements.

### Data Quality & Validation Gates (`validate_gold`)
Fails immediately and raises `ValueError` if:
- Dataset is empty.
- Any of the 18 required columns are missing.
- Duplicate `order_id` values remain.
- Any row's `order_date` does not match the batch date partition.
- `late_delivery` contains values other than 0 or 1, or contains `NaN`.

---

## 4. Machine Learning Pipeline (`src/train.py`)

### Separation of Concerns & Leak-Free Preprocessing
To strictly avoid data leakage:
- Preprocessing steps that compute sample statistics (such as median imputation or one-hot categorical encoding) are encapsulated inside a Scikit-learn `Pipeline` and `ColumnTransformer`.
- These transformers are fit **strictly on training data** (`X_train`) and only transform `X_test`.

### Pipeline Structure
- **Numerical Pipeline**: `SimpleImputer(strategy="median")`
- **Categorical Pipeline**: `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`
- **Classifier**: `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")`

### Artifacts Generated
- Model: `models/delivery_delay_model.joblib`
- Metrics: `metrics/metrics.json`

Example `metrics/metrics.json`:
```json
{
  "model_name": "RandomForestClassifier",
  "split_strategy": "stratified_split_80_20",
  "total_records": 798,
  "train_records": 638,
  "test_records": 160,
  "date_range": ["2026-10-01", "2026-10-20"],
  "metrics": {
    "accuracy": 0.7312,
    "precision": 0.4324,
    "recall": 0.4211,
    "f1_score": 0.4267
  }
}
```

---

## 5. Airflow DAG Orchestration

The project implements two dedicated DAGs in `dags/`:

### 1. `data_pipeline` (`dags/data_pipeline.py`)
- **Schedule**: `@daily`
- **Retries**: `retries=2`, `retry_delay=timedelta(seconds=10)`
- **Date Parameter**: Uses logical date `{{ ds }}`
- **Task Order**:
  $$\text{extract\_to\_bronze} \longrightarrow \text{transform\_to\_silver} \longrightarrow \text{prepare\_gold} \longrightarrow \text{validate\_gold}$$

### 2. `ml_training_pipeline` (`dags/ml_training_pipeline.py`)
- **Schedule**: `None` (manual trigger)
- **Task Order**:
  $$\text{load\_gold\_data} \longrightarrow \text{train\_model} \longrightarrow \text{evaluate\_model} \longrightarrow \text{save\_model}$$

---

## 6. Project Structure

```
assignment04-delivery-delay/
├── dags/
│   ├── data_pipeline.py           # Daily Medallion Airflow DAG
│   └── ml_training_pipeline.py    # ML Training Airflow DAG
├── src/
│   ├── etl.py                     # Bronze, Silver, Gold, and Validation logic
│   └── train.py                   # Leak-free training, evaluation, and serialization
├── data/
│   ├── raw/                       # 46 immutable daily CSVs (orders_YYYY-MM-DD.csv)
│   ├── bronze/                    # Verbatim daily raw copies
│   ├── silver/                    # Cleaned and deduplicated daily partitions
│   ├── gold/                      # Feature-engineered daily partitions
│   └── ml/                        # Staged training and testing sets
├── models/
│   └── delivery_delay_model.joblib # Persisted Scikit-learn Pipeline
├── metrics/
│   └── metrics.json               # Evaluation metrics and dataset split metadata
├── screenshots/                   # Verification evidence screenshots
├── Dockerfile                     # Custom Airflow image with scikit-learn & pandas
├── docker-compose.yml             # Airflow webserver, scheduler, postgres & init
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusions
├── data_dictionary.csv            # Comprehensive schema dictionary
└── README.md                      # Complete project documentation
```

---

## 7. Setup & Execution Commands

### Local Environment Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test ETL pipeline standalone for a specific date
python src/etl.py --date 2026-10-18 --step all

# 4. Test ML training pipeline standalone
python src/train.py
```

### Docker & Airflow Execution
```bash
# 1. Prepare environment variables
cp .env.example .env

# 2. Build and start Airflow services
docker compose up --build -d

# 3. Check container status
docker compose ps

# 4. Access Airflow Web UI
# Open http://localhost:8080 (User: airflow, Password: airflow)

# 5. Stop Airflow services
docker compose down
```

---

## 8. Reliability Testing & Verification

### 1. Retry Verification
- **Mechanism**: `dags/data_pipeline.py` integrates a transient simulation hook (`SIMULATE_RETRY_ON_DATE`).
- **Behavior**: On attempt 1, the task raises an error; Airflow automatically triggers retry 1 (`try_number = 2`) which executes successfully.

### 2. Idempotency Verification
- **Mechanism**: Running `python src/etl.py --date 2026-10-18 --step all` multiple times.
- **Result**: Row count remains exactly 245 rows. Deduplication on `order_id` ensures zero duplicate orders are introduced upon repeated runs.

### 3. Backfill Verification
- Executed on required three dates: `2026-10-18`, `2026-10-19`, and `2026-10-20`.
- Verified outputs:
  - `data/gold/orders_2026-10-18.csv` (245 validated rows)
  - `data/gold/orders_2026-10-19.csv` (180 validated rows)
  - `data/gold/orders_2026-10-20.csv` (193 validated rows)

