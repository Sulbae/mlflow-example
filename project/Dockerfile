# Base image Python
FROM python:3.12.12-slim

# Working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.14.3 \
    mlflow==2.14.3 \
    cloudpickle==3.1.1 \
    numpy==1.26.4 \
    packaging==24.2 \
    pandas==2.3.3 \
    pyarrow==15.0.2 \
    pyyaml==6.0.3 \
    scikit-learn==1.7.1 \
    scipy==1.16.2 \
    dagshub==0.6.3
