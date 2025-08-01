import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using SMOTE or random undersampling.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        method (str): 'smote' for SMOTE, 'undersample' for random undersampling, or 'none'.
        random_state (int): Random seed.
    Returns:
        X_res, y_res: Resampled feature matrix and target vector.
    """
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X, y)
    elif method == 'none':
        X_res, y_res = X, y
    else:
        raise ValueError("Unknown method for class imbalance handling.")
    return X_res, y_res


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler.
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        method (str): 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
    Returns:
        X_train_scaled, X_test_scaled: Scaled feature matrices.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaling method.")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def encode_categorical_features(X_train, X_test, categorical_cols, max_categories=50):
    """
    One-hot encode categorical features in train and test sets, aligning columns.
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        categorical_cols (list): List of categorical column names.
        max_categories (int): Maximum number of categories to keep per column (default 50).
    Returns:
        X_train_encoded, X_test_encoded: Encoded and aligned feature matrices.
    """
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    # Handle high-cardinality categorical columns
    for col in categorical_cols:
        if col in X_train_copy.columns:
            # Get top categories from training data
            top_categories = X_train_copy[col].value_counts().head(max_categories).index.tolist()
            
            # Replace rare categories with 'Other'
            X_train_copy[col] = X_train_copy[col].apply(lambda x: x if x in top_categories else 'Other')
            X_test_copy[col] = X_test_copy[col].apply(lambda x: x if x in top_categories else 'Other')
            
            print(f"Column '{col}': kept top {len(top_categories)} categories, others grouped as 'Other'")
    
    X_train_enc = pd.get_dummies(X_train_copy, columns=categorical_cols)
    X_test_enc = pd.get_dummies(X_test_copy, columns=categorical_cols)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)
    return X_train_enc, X_test_enc 