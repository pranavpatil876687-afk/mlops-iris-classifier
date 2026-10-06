import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

# Target column
TARGET = "species"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Keep only numerical features
X = X.select_dtypes(include="number")


# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 3. Baseline model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)


# -----------------------------
# 4. Cross-validation
# -----------------------------
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1_macro"
)

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()


# -----------------------------
# 5. Train final model
# -----------------------------
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)


# -----------------------------
# 6. MLflow tracking
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-hyperparameter-tuning")

with mlflow.start_run(run_name="baseline_decision_tree"):

    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("cv_f1_macro", cv_mean)
    mlflow.log_metric("cv_f1_std", cv_std)
    mlflow.log_metric("test_accuracy", test_accuracy)

    mlflow.sklearn.log_model(
        model,
        "model"
    )

print(f"Baseline CV f1_macro: {cv_mean:.4f} (+/- {cv_std:.4f})")
print(f"Baseline test accuracy: {test_accuracy:.4f}")