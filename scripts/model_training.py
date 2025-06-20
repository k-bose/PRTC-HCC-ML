import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, confusion_matrix

# --- Custom Scorer for Specificity ---
def specificity_score(y_true, y_pred):
    """
    Calculates specificity.
    Specificity = True Negatives / (True Negatives + False Positives)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

# Define the scorers we want to use
# Note: Sensitivity is called 'recall' in scikit-learn
scoring_metrics = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'recall': 'recall', # This is Sensitivity
    'specificity': make_scorer(specificity_score) # Our custom scorer
}

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
    # Load Config and Data
    config = get_config()
    X, y = load_ds(config['DATASET'])

    # Define the cross-validation strategy
    k_folds = 5
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['RANDOM_STATE'])

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=config['RANDOM_STATE']),
        # Increased max_iter to help with convergence on high-dimensional data
        'Logistic Regression': LogisticRegression(random_state=config['RANDOM_STATE'], max_iter=1000),
        'SVM': SVC(probability=True, random_state=config['RANDOM_STATE']),
        'Gradient Boosting': GradientBoostingClassifier(random_state=config['RANDOM_STATE']),
        # Added parameters to avoid deprecation warnings and set a default eval metric
        'XGB': XGBClassifier(random_state=config['RANDOM_STATE'], use_label_encoder=False, eval_metric='logloss')
    }

    print(f"--- Evaluating models with {k_folds}-fold Cross-Validation ---")

    # Evaluate each model using a pipeline
    for name, model in models.items():
        # Create a pipeline that first scales the data, then fits the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Use cross_validate to get multiple scores at once
        # It handles fitting the scaler on train and transforming test within each fold
        cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring_metrics)

        # Print the results for the current model
        print(f"\n--- {name} ---")
        print(f"Accuracy:    {np.mean(cv_results['test_accuracy']):.4f} (std: {np.std(cv_results['test_accuracy']):.4f})")
        print(f"F1 Score:    {np.mean(cv_results['test_f1']):.4f} (std: {np.std(cv_results['test_f1']):.4f})")
        print(f"AUC Score:   {np.mean(cv_results['test_roc_auc']):.4f} (std: {np.std(cv_results['test_roc_auc']):.4f})")
        print(f"Sensitivity: {np.mean(cv_results['test_recall']):.4f} (std: {np.std(cv_results['test_recall']):.4f}) (Recall)")
        print(f"Specificity: {np.mean(cv_results['test_specificity']):.4f} (std: {np.std(cv_results['test_specificity']):.4f})")


if __name__ == "__main__":
    main()