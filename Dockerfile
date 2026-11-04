FROM apache/airflow:2.9.2-python3.10

USER root
# Install curl and procps for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

