import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint


# --------------------------------
# 1. Load dataset
# --------------------------------
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "species"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Use numerical features only
X = X.select_dtypes(include="number")


# --------------------------------
# 2. Train-test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 3. Random Forest
# --------------------------------
model = RandomForestClassifier(
    random_state=42
)


# --------------------------------
# 4. Parameter distributions
# --------------------------------
param_distributions = {
    "n_estimators": randint(50, 301),
    "max_depth": [3, 5, 10, None],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2"]
}


# --------------------------------
# 5. Random Search
# --------------------------------
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    return_train_score=False
)


print("Starting Random Search...")
print("Random combinations: 30")
print("Total fits: 150")

random_search.fit(X_train, y_train)


# --------------------------------
# 6. Best model
# --------------------------------
best_model = random_search.best_estimator_

best_cv_score = random_search.best_score_

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)


# --------------------------------
# 7. MLflow tracking
# --------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="random_search_random_forest"):

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "search_method",
        "RandomizedSearchCV"
    )

    mlflow.log_param(
        "cv_folds",
        5
    )

    mlflow.log_param(
        "random_combinations",
        30
    )

    mlflow.log_param(
        "total_fits",
        150
    )

    # Log best parameters
    for parameter, value in random_search.best_params_.items():
        mlflow.log_param(
            f"best_{parameter}",
            value
        )

    # Log metrics
    mlflow.log_metric(
        "best_cv_f1_macro",
        best_cv_score
    )

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    # Log model
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )


# --------------------------------
# 8. Display results
# --------------------------------
print("\nRandom Search completed!")

print("\nBest parameters:")
print(random_search.best_params_)

print(f"\nBest CV f1_macro: {best_cv_score:.4f}")

print(f"Test accuracy: {test_accuracy:.4f}")

print("\nTotal fits: 150")