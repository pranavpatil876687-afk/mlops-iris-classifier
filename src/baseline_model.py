
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def run_baseline():
    df = pd.read_csv("data/processed/iris_features.csv")

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = LabelEncoder().fit_transform(df["species"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    with mlflow.start_run(run_name="baseline_decision_tree"):
        model = DecisionTreeClassifier(random_state=42)

        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="f1_macro"
        )

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", "DecisionTreeClassifier_default")
        mlflow.log_metric("cv_f1_macro_mean", scores.mean())
        mlflow.log_metric("cv_f1_macro_std", scores.std())
        mlflow.log_metric("test_accuracy", accuracy)

        print(f"CV F1 Macro: {scores.mean():.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run_baseline()
