import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "species"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Use numerical features only
X = X.select_dtypes(include="number")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest model
model = RandomForestClassifier(
    random_state=42
)


# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}


# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    return_train_score=False
)


print("Starting Grid Search...")
print("Total combinations:", 72)
print("Total fits:", 72 * 5)

grid_search.fit(X_train, y_train)


# Best model
best_model = grid_search.best_estimator_

best_cv_score = grid_search.best_score_

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)


# MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="grid_search_random_forest"):

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("search_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("total_combinations", 72)
    mlflow.log_param("total_fits", 360)

    for parameter, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{parameter}", value)

    mlflow.log_metric("best_cv_f1_macro", best_cv_score)
    mlflow.log_metric("test_accuracy", test_accuracy)

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )


print("\nGrid Search completed!")
print("Best parameters:")
print(grid_search.best_params_)

print(f"\nBest CV f1_macro: {best_cv_score:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("\nTotal fits: 360")