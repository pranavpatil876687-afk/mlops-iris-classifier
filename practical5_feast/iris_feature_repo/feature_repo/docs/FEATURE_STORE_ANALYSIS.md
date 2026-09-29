# Feature Store Analysis

## 1. Centralized Feature Management

Feast provides a centralized repository for defining and managing ML features.

In this experiment, the Iris features were registered using:

- sample_id as the entity
- iris_measurements as the original feature view
- iris_engineered_features as the engineered feature view

This avoids defining the same features separately in different ML applications.

## 2. Online and Offline Feature Retrieval

The Feature Store supports two different retrieval patterns.

### Online Retrieval

The online store is used for low-latency feature retrieval during inference.

In this experiment, SQLite was used as the local online store.

The online retrieval script successfully retrieved features for sample_id = 1.

### Offline Retrieval

Historical features were retrieved using Feast's historical feature retrieval.

The event_timestamp allowed Feast to retrieve feature values corresponding to the appropriate point in time.

## 3. Point-in-Time Correctness

Historical feature retrieval helps prevent future information from being used when constructing training data.

This reduces the risk of future-data leakage during model training.

## 4. Feature Reusability

The registered features were reused through the iris_feature_service.

The same registered feature definitions can be consumed by different ML tasks without independently reimplementing the feature calculations.

## 5. Reduction of Training-Serving Skew

The same feature definitions are used for online and offline retrieval.

This helps ensure that the features used during training and inference are consistent.

## 6. Benefits Observed

The experiment demonstrated the following benefits:

- Centralized feature definitions
- Reusable ML features
- Online feature retrieval
- Historical/offline feature retrieval
- Point-in-time correctness
- Reduced training-serving skew
- Consistent feature definitions across different ML models