import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

class ModelWrapper:
    
    def __init__(self, model_dict):
        self.trees = model_dict['trees']
        self.task_type = model_dict['task_type']
        self.learning_rate = model_dict.get('learning_rate', 0.1)
        
        if self.task_type == "multiclass":
            self.n_classes = len(np.unique(model_dict['label']))
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.reset_index(drop=True)
        
        if self.task_type == "regression":
            pred = np.zeros(len(X))
        elif self.task_type == "binary":
            pred = np.zeros(len(X))
        elif self.task_type == "multiclass":
            pred = np.zeros((len(X), self.n_classes))
        
        for tree in self.trees:
            for i in range(len(X)):
                sample = X.iloc[i:i+1] if isinstance(X, pd.DataFrame) else X[i:i+1]
                tree_pred = self._predict_single_tree(tree, sample)
                pred[i] += self.learning_rate * tree_pred
        
        if self.task_type == "regression":
            return pred
        elif self.task_type == "binary":
            return (sigmoid(pred) > 0.5).astype(int)
        elif self.task_type == "multiclass":
            return np.argmax(softmax(pred), axis=1)
    
    def _predict_single_tree(self, node, X):
        if node.left is None and node.right is None:
            return node.value
        
        feature_val = X[node.feature_name].iloc[0] if isinstance(X, pd.DataFrame) else X[node.feature_name]
        if feature_val < node.threshold:
            return self._predict_single_tree(node.left, X)
        else:
            return self._predict_single_tree(node.right, X)
    
    def predict_proba(self, X):
        if self.task_type == "regression":
            raise ValueError("Probability prediction not supported for regression")
        
        if self.task_type == "binary":
            logits = np.zeros(len(X))
        else:
            logits = np.zeros((len(X), self.n_classes))
        
        for tree in self.trees:
            for i in range(len(X)):
                sample = X.iloc[i:i+1] if isinstance(X, pd.DataFrame) else X[i:i+1]
                logits[i] += self.learning_rate * self._predict_single_tree(tree, sample)
        
        if self.task_type == "binary":
            proba = sigmoid(logits)
            return np.column_stack([1-proba, proba])
        else:
            return softmax(logits)