import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

def get_sample_training_data():
    """
    Return a small sample training dataset for unit testing.
    """
    X_train = np.array([
        [25, 40],
        [35, 45],
        [45, 50],
        [55, 60],
    ])
    y_train = np.array([0, 0, 1, 1])

    return X_train, y_train


# Helper function tests

def test_apply_label():
    """
    Verify that binary predictions are converted to the expected salary labels.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


# Model training tests

def test_train_model_returns_random_forest():
    """
    Verify that train_model returns a fitted RandomForestClassifier.
    """
    X_train, y_train = get_sample_training_data()
   
    model = train_model(X_train, y_train)
   
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_perfect_predictions():
    """
    Verify that compute_model_metrics returns perfect scores for perfect predictions.
    """
    y_true = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_save_and_load_model(tmp_path):
    """
    Verify that a trained model can be saved and loaded successfully.
    """
    X_train, y_train = get_sample_training_data()

    model = train_model(X_train, y_train)
    model_path = tmp_path / "test_model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, RandomForestClassifier)
    

def test_inference_returns_expected_number_of_predictions():
    """
    Verify that inference returns one prediction for each input row.
    """
    X_train, y_train = get_sample_training_data()

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(X_train)


# Data preprocessing tests

def test_process_data_returns_expected_shapes():
    """
    Verify that process_data returns feature and label arrays
    with the expected number of rows.
    """
    data = pd.DataFrame({
        "age": [25, 35, 45],
        "workclass": ["Private", "State-gov", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales"],
        "relationship": ["Not-in-family", "Husband", "Unmarried"],
        "race": ["White", "White", "Black"],
        "sex": ["Male", "Female", "Male"],
        "native-country": ["United-States", "United-States", "Canada"],
        "salary": ["<=50K", ">50K", "<=50K"],
    })

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == 3
    assert y.shape[0] == 3