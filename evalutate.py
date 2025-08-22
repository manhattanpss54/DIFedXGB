import pickle
import os
import pandas as pd
from utils.modelwrapper import ModelWrapper
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_model(y_true, y_pred, task_type):
    if task_type == "regression":
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
    else:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'saved_models', 'difedxgb_model.pkl')
loaded_model_dict = load_model(path)
predictor = ModelWrapper(loaded_model_dict)

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'dataset', 'adult.data')
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

predictions = predictor.predict(X)
eval_results = evaluate_model(y, predictions, predictor.task_type)