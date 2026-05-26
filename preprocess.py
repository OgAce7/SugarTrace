import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Loading data :>
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataset loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def fix_zero_values(df):
# Handles missing values encoded as zeros. Biological data like 'Glucose' or 'BMI' cannot logically be 0, so we treat them as missing and impute using the median to stay robust against outliers.
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nZero counts before fixing:")
    for col in cols_with_zeros:
        zeros = (df[col] == 0).sum()
        print(f"  {col}: {zeros} zeros")
        
    # Replace 0s with NaN to facilitate proper column-wise imputation
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # Impute missing values with the median
    for col in cols_with_zeros:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {col} NaNs with median: {median_val:.2f}")
    return df

# Separates target variable from input features :x
def split_features_labels(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    print(f"\nFeatures shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    return X, y

# Standardizes features by removing the mean and scaling to unit variance
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Serializes the fitted scaler for consistent transform during inference :p
def save_scaler(scaler, path='models/scaler.pkl'):
    os.makedirs('models', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")

# Loads the serialized scaler
def load_scaler(path='models/scaler.pkl'):
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


# just for testing this file alone
if __name__ == "__main__":
    # this won't run unless we call the file directly
    print("preprocess.py loaded successfully")
