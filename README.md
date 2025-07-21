# Fraud Detection for E-commerce and Banking

A comprehensive machine learning project for detecting fraudulent transactions in e-commerce and banking datasets. This project implements advanced preprocessing techniques, handles class imbalance, incorporates geolocation analysis, and provides modular, reusable code for fraud detection systems.

## 📊 Project Overview

This project addresses fraud detection challenges across two domains:
- **E-commerce transactions** with geolocation mapping
- **Banking credit card transactions** 
- **Class imbalance handling** (fraud is typically <10% of data)
- **Feature engineering** for transaction patterns
- **Model explainability** using SHAP

## 🗂️ Project Structure

```
Fraud_Detection_For_Ecommerce_and_Banking/
├── data/
│   ├── raw/                    # Raw datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/              # Preprocessed data outputs
├── src/
│   └── utils/                  # Modular utility functions
│       ├── data_cleaning.py    # Missing values, duplicates, data types
│       ├── eda.py              # Exploratory data analysis functions
│       ├── feature_engineering.py  # IP conversion, time features, frequencies
│       └── data_transformation.py  # SMOTE, scaling, encoding
├── notebooks/
│   └── 01_task1_data_preprocessing.ipynb  # Complete Task 1 implementation
├── config/                     # Configuration files
├── tests/                      # Unit and integration tests
└── requirements.txt           # Project dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Fraud_Detection_For_Ecommerce_and_Banking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place your datasets in the `data/raw/` directory:
- `Fraud_Data.csv` - E-commerce transaction data
- `IpAddress_to_Country.csv` - IP geolocation mapping
- `creditcard.csv` - Banking transaction data

### 3. Run Task 1: Data Preprocessing

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_task1_data_preprocessing.ipynb
```

## 📋 Task 1: Data Analysis and Preprocessing ✅

### What's Implemented

Our Task 1 provides a **complete, production-ready preprocessing pipeline**:

#### 🧹 **Data Cleaning**
- **Missing Value Handling**: Configurable strategies (mean/median/mode/drop)
- **Duplicate Removal**: Automatic detection and removal
- **Data Type Correction**: Proper datetime, numeric, and categorical types

#### 📊 **Exploratory Data Analysis (EDA)**
- **Univariate Analysis**: Distribution plots for all features
- **Bivariate Analysis**: Feature-target relationship visualization
- **Summary Statistics**: Comprehensive data profiling
- **Class Imbalance Visualization**: Critical for fraud detection

#### 🌍 **Geolocation Integration**
- **IP Address Conversion**: Scientific notation to proper IP format
- **Country Mapping**: Merge transactions with geographical data
- **Location-based Features**: Country-level fraud analysis

#### ⚙️ **Feature Engineering**
- **Time-based Features**: Hour of day, day of week, time since signup
- **Transaction Frequency**: User and device transaction counts
- **Geolocation Features**: Country-based risk indicators

#### ⚖️ **Data Transformation**
- **Class Imbalance Handling**: SMOTE for balanced training data
- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: Smart one-hot encoding with cardinality limits
- **Memory Optimization**: Handles high-cardinality categorical data

### Key Features

✅ **Memory Efficient**: Handles large datasets with cardinality limits  
✅ **Modular Design**: Reusable functions for any fraud detection project  
✅ **Production Ready**: Error handling and data validation  
✅ **Well Documented**: Step-by-step explanations for beginners  
✅ **Configurable**: Easy to adapt for different datasets  

## 🛠️ Modular Functions

### Data Cleaning (`src/utils/data_cleaning.py`)
```python
from src.utils.data_cleaning import handle_missing_values, remove_duplicates, correct_data_types

# Handle missing values with custom strategies
clean_df = handle_missing_values(df, {'age': 'median', 'browser': 'mode'})

# Remove duplicates
clean_df = remove_duplicates(clean_df)

# Correct data types
clean_df = correct_data_types(clean_df, {'signup_time': 'datetime64[ns]'})
```

### Feature Engineering (`src/utils/feature_engineering.py`)
```python
from src.utils.feature_engineering import add_time_features, add_transaction_frequency

# Add time-based features
df = add_time_features(df, 'purchase_time', 'signup_time')

# Add frequency features
df = add_transaction_frequency(df, 'user_id', 'user_txn_count')
```

### Data Transformation (`src/utils/data_transformation.py`)
```python
from src.utils.data_transformation import handle_class_imbalance, encode_categorical_features

# Handle class imbalance with SMOTE
X_res, y_res = handle_class_imbalance(X, y, method='smote')

# Encode categorical features (with cardinality limits)
X_train_enc, X_test_enc = encode_categorical_features(X_train, X_test, categorical_cols, max_categories=50)
```

## 📈 Results Summary

After running Task 1, you'll have:

- **Clean, structured data** ready for modeling
- **Balanced dataset** (fraud vs non-fraud)
- **Rich feature set** with time, frequency, and location features
- **Properly scaled and encoded** features
- **Memory-efficient processing** of high-cardinality data

### Sample Output:
```
=== PREPROCESSING COMPLETE ===
Final training features shape: (218,000, 75)  # Balanced with SMOTE
Final test features shape: (30,223, 75)
Training class balance: {0: 109568, 1: 109568}  # Perfect balance
✅ Data is ready for Task 2: Model Building and Training!
```

## 🔧 Configuration

Customize the preprocessing pipeline by modifying these variables in the notebook:

```python
# Missing value strategies
missing_strategy = {
    'age': 'median',
    'browser': 'mode',
    # Add your columns here
}

# Feature columns to use
feature_columns = [
    'user_id', 'purchase_value', 'age', 
    'browser', 'source', 'sex', 'country',
    # Add your features here
]

# Categorical encoding limits
max_categories = 50  # Adjust based on your memory constraints
```

## 🚨 Common Issues and Solutions

### Memory Error during Encoding
**Problem**: `MemoryError: Unable to allocate X GB`  
**Solution**: Reduce `max_categories` parameter or remove high-cardinality columns

### IP Address Format Issues
**Problem**: IP addresses in scientific notation  
**Solution**: Our pipeline automatically handles conversion

### SMOTE Fails
**Problem**: SMOTE requires numeric data  
**Solution**: Ensure categorical encoding runs first (our pipeline handles this)

## 🎯 Next Steps: Task 2 & 3

After completing Task 1, you'll be ready for:

- **Task 2**: Model Building and Training (Logistic Regression, Random Forest, XGBoost)
- **Task 3**: Model Explainability (SHAP analysis, feature importance)

## 📚 Dependencies

Key libraries used:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `imbalanced-learn` - SMOTE for class balancing
- `matplotlib`, `seaborn` - Visualization
- `ipaddress` - IP address handling

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
1. Fork the repository
2. Add your own feature engineering functions
3. Experiment with different preprocessing strategies
4. Submit pull requests with improvements

## 📄 License

This project is for educational use in AI/ML courses.

---

**🎓 Perfect for**: Data Science students, ML practitioners, fraud detection projects  
**📊 Handles**: Class imbalance, high-cardinality data, geolocation, time series features  
**⚡ Ready for**: Production deployment, academic research, portfolio projects
