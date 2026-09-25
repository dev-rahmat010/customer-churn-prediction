import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def generate_data(n_samples=5000):
    """Generates synthetic customer data for churn prediction."""
    np.random.seed(42)
    data = {
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20.0, 120.0, n_samples),
        'total_charges': lambda df: df['tenure_months'] * df['monthly_charges'] + np.random.normal(0, 50, n_samples),
        'support_tickets': np.random.randint(0, 5, n_samples),
        'contract_type': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]), # 0: Month-to-month, 1: One year, 2: Two year
        'is_senior': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    df['total_charges'] = df['tenure_months'] * df['monthly_charges']
    
    # Target variable: Churn logic based on rules + noise
    churn_prob = (
        (df['contract_type'] == 0) * 0.4 +
        (df['support_tickets'] > 2) * 0.3 +
        (df['monthly_charges'] > 90) * 0.2 -
        (df['tenure_months'] > 24) * 0.3
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churn'] = np.random.binomial(1, churn_prob)
    
    return df

def main():
    print("Generating dataset...")
    df = generate_data()
    
    # Features and Target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
    
    # Save model
    joblib.dump(model, 'churn_model.pkl')
    print("\nModel saved successfully as 'churn_model.pkl'!")

if __name__ == "__main__":
    main()
