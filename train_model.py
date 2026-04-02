# Loading data and model training script
import kagglehub
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

# importing preprocessing functions
from preprocess import load_data, fix_zero_values, split_features_labels, scale_features, save_scaler

print("=" * 50)
print("SugarTrace - Diabetes Prediction Model Training")
print("=" * 50)

# Loading dataset
print("\n[Step 1] Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

csv_file = None
for f in os.listdir(path):
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break

if csv_file is None:
    raise FileNotFoundError("Could not find CSV file in dataset path")

print(f"Found CSV: {csv_file}")
df = load_data(csv_file)

# Data cleaning
print("\n[Step 2] Cleaning data (fixing zero values)...")
df = fix_zero_values(df)

# Data splitting
print("\n[Step 3] Splitting features and labels...")
X, y = split_features_labels(df)

feature_names = X.columns.tolist()
print("Feature names:", feature_names)

# Training and Testing
print("\n[Step 4] Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Feature Scaling
print("\n[Step 5] Scaling features...")
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
save_scaler(scaler)

# LR model training
print("\n[Step 6] Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_probs)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy : {lr_accuracy:.4f}")
print(f"Recall   : {lr_recall:.4f}")
print(f"ROC-AUC  : {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_preds))

# RF model training
print("\n[Step 7] Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)

print("\n--- Random Forest Results ---")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"ROC-AUC  : {rf_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_preds))

# Feature importance
print("\n[Step 8] Feature Importances (Random Forest):")
importances = rf_model.feature_importances_
feature_importance_dict = dict(zip(feature_names, importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_features:
    print(f"  {feat}: {imp:.4f}")

# Saving models
print("\n[Step 9] Saving models...")
os.makedirs('models', exist_ok=True)

with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("Logistic Regression model saved.")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Random Forest model saved.")

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("Feature names saved.")

# Results
results = {
    'logistic_regression': {
        'accuracy': lr_accuracy,
        'recall': lr_recall,
        'roc_auc': lr_auc
    },
    'random_forest': {
        'accuracy': rf_accuracy,
        'recall': rf_recall,
        'roc_auc': rf_auc
    },
    'feature_importance': feature_importance_dict
}

with open('models/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 50)
print("Training complete! All models saved in /models folder.")
print("=" * 50)
