import pandas as pd
from A import A
from B import B
from C import C
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.activation import sigmoid
from sklearn.preprocessing import LabelEncoder
from utils.model_evaluate import ModelEvaluator
import pickle

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'dataset', 'adult.data')
    
    dataset = pd.read_csv(path, header=None, skipinitialspace=True)
    FEATURES = dataset.columns.tolist()
    LABEL = FEATURES[-1]
    FEATURES = FEATURES[:-1]
    
    FEATURES_A = FEATURES[:7]
    FEATURES_B = FEATURES[7:]
    DEPTH = 3

    a = A(global_features=FEATURES, local_features=FEATURES_A, label=LABEL)
    b = B(global_features=FEATURES, local_features=FEATURES_B, label=LABEL)
    c = C()

    while True:
        result = a.upload_gh()
        if result is None:
            pass
        else:
            gradients, hessians = result
            c.set_gh(gradients, hessians)

        a_res = a.upload_inf()
        b_res = b.upload_inf()
        
        feature_name, thr = c.aggregate_inf(a_res[1], b_res[1], a_res[0], b_res[0])

        a.create_node(feature_name, thr)
        b.create_node(feature_name, thr)

        if a._is_done:
            path = os.path.join(current_dir, 'dataset', 'adult.data')
            dataset = pd.read_csv(path, header=None, names=FEATURES+[LABEL], skipinitialspace=True)

            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]

            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

            a.evaluate(X=X, y_true=y)
            break

if __name__ == "__main__":
    main()