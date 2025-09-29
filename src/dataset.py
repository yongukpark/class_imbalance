from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# ===== dataset list =====

class TabularDataset:
    dataset_1 = [40983, 38, 1068, 40994, 1487]  # severe imbalance
    dataset_2 = [1050, 1461, 1049, 40978, 40701]
    dataset_3 = [1067, 1063, 1590, 1053, 1464]
    dataset_4 = [1486, 1480, 1489, 31] # little imbalance
    dataset_total = dataset_1 + dataset_2 + dataset_3 + dataset_4


def load_and_encode_dataset(data_id):

    # data load
    data = fetch_openml(data_id=data_id, as_frame=True)
    X_raw = data.data
    y_raw = data.target

    numeric_features = X_raw.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_features = X_raw.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X = preprocessor.fit_transform(X_raw)

    # normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    
    # minority class to "label 1"
    class_counts = np.bincount(y_encoded)
    minority_class = np.argmin(class_counts)
    y = np.where(y_encoded == minority_class, 1, 0)

    X = pd.DataFrame(X)

    return X, y
