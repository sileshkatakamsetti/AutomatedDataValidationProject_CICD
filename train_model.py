import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_and_prepare_data(csv_path: str):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Simple encoding: convert all object (string) columns to numeric labels
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Split features (X) and target (y)
    if "income" not in df.columns:
        raise ValueError("Column 'income' not found in dataset. This will be the target variable.")

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y, encoders

def train_and_evaluate(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    return model, acc

def main():
    csv_path = "data/census_sample.csv"  # same file used in validation

    print("[MODEL] Loading and preparing data...")
    X, y, encoders = load_and_prepare_data(csv_path)

    print("[MODEL] Training Logistic Regression model...")
    model, acc = train_and_evaluate(X, y)

    # Save model & encoders for future use
    joblib.dump(model, "income_model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    print("[MODEL] Saved model to income_model.pkl and encoders to encoders.pkl")

if __name__ == "__main__":
    main()
