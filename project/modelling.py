import mlflow
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

# Konfigurasi Parameter
DATASET_PATH = "train_pca.csv"
N_ESTIMATOR = 100
MAX_DEPTH = 5

# Set Tracking melalui DagsHub
dagshub.init(
    repo_owner="Sulbae",
    repo_name="Latihan-MLFlow",
    mlflow=True
)

mlflow.set_experiment("Latihan Credit Scoring")
# Aktifkan autolog
mlflow.autolog(log_models=False)

data = pd.read_csv(DATASET_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Credit_Score", axis=1), 
    data["Credit_Score"], 
    test_size=0.2, 
    random_state=42
)

input_example = X_train.iloc[0:5]

with mlflow.start_run():
    
    # Log parameters
    mlflow.log_param("n_estimators", N_ESTIMATOR)
    mlflow.log_param("max_depth", MAX_DEPTH)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATOR,
        max_depth=MAX_DEPTH
    )
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )