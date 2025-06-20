import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculates and prints a comprehensive set of evaluation metrics for the model.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Calculate Core Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # --- 3. Print Results ---
    print(f"--- Evaluation Metrics for {model_name} ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC Score:   {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}\n")


def preprocess_data(X_train, X_test):
    # Standardizing
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Transform data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Recreate DataFrames preserving original index to prevent misalignment with y_train/y_test
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled


def load_ds(ds):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(ds)
    print(f"Total Rows x Columns in Data: {df.shape}\n")

    target_column = 'initial_LR_TR'
    y = df[target_column]
    X = df.drop(columns=[target_column])

    return X, y


def get_config():
    """Returns a dictionary of configuration parameters."""
    return {
        'DATASET': 'data/radiomics_ct_features_cleaned.csv',
        'RANDOM_STATE': 42
    }


def main():
    # Load Config
    config = get_config()

    # Load Dataset
    X, y = load_ds(config['DATASET'])

    # 90-10 Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        random_state=config['RANDOM_STATE'], 
        stratify=y
    )
    print(f"No. of training samples (X_train): {X_train.shape[0]}")
    print(f"No. of testing samples (X_test): {X_test.shape[0]}\n")

    # Preprocessing
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)  

    # Define Models
    models = {
        'Random Forest': RandomForestClassifier(random_state=config['RANDOM_STATE']),
        'SVM': SVC(probability=True, random_state=config['RANDOM_STATE']),
        'Logistic Regression': LogisticRegression(random_state=config['RANDOM_STATE']),
        'Gradient Boosting': GradientBoostingClassifier(random_state=config['RANDOM_STATE']),
        'XGB': XGBClassifier(random_state=config['RANDOM_STATE'])
    }
    
    # Training
    for name, model in models.items():
        # Training
        print(f"Training the {name} model...")
        model.fit(X_train_scaled, y_train)
        print("Training complete.\n")

        # Evaluation
        evaluate_model(model, X_test_scaled, y_test, name)

    
if __name__ == "__main__":
    main()