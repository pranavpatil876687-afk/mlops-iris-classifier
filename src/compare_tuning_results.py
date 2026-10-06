import pandas as pd


# Results from the three experiments
results = pd.DataFrame({
    "Method": [
        "Baseline Decision Tree",
        "Grid Search Random Forest",
        "Random Search Random Forest"
    ],
    "CV F1 Macro": [
        0.9665,
        0.9663,
        0.9580
    ],
    "Test Accuracy": [
        0.9000,
        0.9667,
        0.9667
    ]
})


# Display comparison
print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results.to_string(index=False))


# Find best model based on test accuracy
best_accuracy = results.loc[
    results["Test Accuracy"].idxmax()
]

print("\n========================================")
print("BEST MODEL BY TEST ACCURACY")
print("========================================")

print(f"Method: {best_accuracy['Method']}")
print(f"Test Accuracy: {best_accuracy['Test Accuracy']:.4f}")


# Find best model based on CV F1
best_f1 = results.loc[
    results["CV F1 Macro"].idxmax()
]

print("\n========================================")
print("BEST MODEL BY CV F1")
print("========================================")

print(f"Method: {best_f1['Method']}")
print(f"CV F1 Macro: {best_f1['CV F1 Macro']:.4f}")