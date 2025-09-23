import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    # Features & labels
    X = df.drop(columns=["subject", "sessionIndex", "rep"]).values
    y = df["subject"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=test_size,
        random_state=random_state,
        stratify=y_categorical
    )

    # Add channel dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, y_categorical.shape[1]
