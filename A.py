from collections import deque
import pandas as pd
import os
from utils.Treenode import Treenode
from utils.infer_task_type import infer_task_type
from utils.create_equal_width_bins import create_equal_width_bins
from utils.activation import sigmoid, softmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, 
    r2_score, confusion_matrix, classification_report
)

class A:
    _is_done = False

    def __init__(self, 
                 global_features=None, local_features=None, 
                 label=None,
                 max_depth=2, n_estimators=2,
                 learning_rate=0.1):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'dataset', 'processed', 'dataset_A.csv')
        self.dataset = pd.read_csv(path)

        self.label = label
        self.y = self.dataset[self.label]
        self.X = self.dataset.drop(self.label, axis=1)

        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)

        self.tree_id = 0
        self.node_id = 0
        self.root = None
        self.trees = []
        self.tree_parent_queue = deque()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.local_features = local_features
        self.global_features = [feature for feature in global_features if feature != self.label]

        self.max_gain = 0
        self.best_feature = None
        self.best_threshold = None

        self.task_type = infer_task_type(self.y)
        self.feature_round = 0
        self.y_pred = None
        self.max_depth = max_depth
        self.depth = 0
        self.new_tree = True

    def upload_gh(self):
        if self.new_tree:
            self.new_tree = False  
            task_type = self.task_type
            y_true = self.y

            if self.y_pred is None:
                self.y_pred = self.initialize_y_pred(self.task_type, y_true)
            else:
                current_tree_pred = self.predict()
                self.y_pred += self.learning_rate * current_tree_pred

            y_pred = self.y_pred
            g, h = self.calculate_gh(task_type, y_true, y_pred)
            self.gh = g, h
            return g, h
        else:
            return None

    def upload_inf(self):
        existence_mask = self.get_existence_mask()
        feature_name_thr_mask = self.get_feature_name_thr_mask()
        return existence_mask, feature_name_thr_mask

    def get_existence_mask(self):
        current_tree_id = self.tree_id
        current_node_id = self.node_id
        
        if self.tree_parent_queue and len(self.tree_parent_queue) > 0:
            parent_node = self.tree_parent_queue[0]
            existence_mask = parent_node.existence_mask
            
            if parent_node.left is None:
                if all(x == 1 for x in existence_mask):
                    existence_mask = existence_mask
                else:
                    existence_mask = [
                        1 if x == 0 else 
                        -1 if x == 1 else 
                        -1 if x == -1 else 
                        x 
                        for x in existence_mask
                    ]
            else:
                if all(x == 1 for x in existence_mask):
                    existence_mask = existence_mask
                existence_mask = [
                    1 if x == 1 else 
                    -1 if x == 0 else 
                    -1 if x == -1 else 
                    x 
                    for x in existence_mask
                ]
        else:
            existence_mask = np.full_like(self.X.iloc[:, 0], 1)
            existence_mask = existence_mask.tolist()
        
        return existence_mask

    def get_feature_name_thr_mask(self):
        feature_name_thr_mask = []
        for feature_name in self.local_features:
            thrs = create_equal_width_bins(feature_data=self.X[feature_name])
            for thr in thrs:
                thr_mask = np.where(self.X[feature_name] < thr, 0, 1)
                feature_name_thr_mask.append([feature_name, thr, thr_mask])
        return feature_name_thr_mask

    def create_node(self, feature_name, thr):
        existence_mask = self.get_existence_mask()

        if feature_name in self.local_features: 
            thr_mask = np.where(self.X[feature_name] < thr, 0, 1)
            new_existence_mask = []
            for e, t in zip(existence_mask, thr_mask):
                if e == 1:
                    new_existence_mask.append(t)
                else:
                    new_existence_mask.append(0)
            existence_mask = new_existence_mask
        else:
            existence_mask = [1 if x == 1 else 0 for x in existence_mask]

        if self.tree_parent_queue and len(self.tree_parent_queue) > 0:
            current_depth = self.tree_parent_queue[0].depth + 1
        else:
            current_depth = 0

        if current_depth > self.max_depth:
            self.tree_parent_queue.clear()
            self._finalize_tree()
            return

        new_node = Treenode(
            tree_id=self.tree_id,
            node_id=self.node_id,
            feature_name=feature_name,
            threshold=thr,
            left=None,
            right=None,
            existence_mask=existence_mask,
            depth=current_depth
        )

        if self.tree_parent_queue and len(self.tree_parent_queue) > 0:
            parent_node = self.tree_parent_queue[0]
            if parent_node.left is None:
                parent_node.left = new_node
            elif parent_node.right is None:
                parent_node.right = new_node
                self.tree_parent_queue.popleft()
        else:
            self.root = new_node

        self.node_id += 1
        self.tree_parent_queue.append(new_node)

    def get_trained_model(self):
        model = {
            'trees': self.trees,
            'task_type': self.task_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'local_features': self.local_features,
            'global_features': self.global_features,
            'label': self.label,
            'feature_importance': self._calculate_feature_importance()
        }
        return model

    def _calculate_feature_importance(self):
        importance = {}
        for tree in self.trees:
            self._traverse_tree_for_importance(tree, importance)
        return importance
    
    def _traverse_tree_for_importance(self, node, importance):
        if node is None or node.feature_name is None:
            return
        if node.feature_name in importance:
            importance[node.feature_name] += 1
        else:
            importance[node.feature_name] = 1
        
        self._traverse_tree_for_importance(node.left, importance)
        self._traverse_tree_for_importance(node.right, importance)

    def initialize_y_pred(self, task_type, y_true):
        if task_type == "regression":
            self.y_pred = np.full_like(y_true, np.mean(y_true), dtype=np.float32)
        elif task_type == "binary":
            self.y_pred = np.zeros_like(y_true, dtype=np.float32)
        elif task_type == "multiclass":
            n_classes = len(np.unique(y_true))
            prior_prob = 1.0 / n_classes
            self.y_pred = np.full((len(y_true), n_classes), np.log(prior_prob), dtype=np.float32)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return self.y_pred

    def calculate_gh(self, task_type, y_true, y_pred):
        if task_type == "regression":
            g = y_pred - y_true
            h = np.ones_like(y_true)
        elif task_type == "binary":
            p = sigmoid(y_pred)
            g = p - y_true
            h = p * (1 - p)
        elif task_type == "multiclass":
            if y_true.ndim == 1:
                y_true_onehot = np.zeros((len(y_true), len(np.unique(y_true))))
                y_true_onehot[np.arange(len(y_true)), y_true] = 1
                y_true = y_true_onehot
            p = softmax(y_pred)
            g = p - y_true
            h = p * (1 - p)
        else:
            raise ValueError("Unsupported task type")
        return g, h

    def _finalize_tree(self):
        g, h = self.gh
        self._assign_leaf_values(self.root, g, h)
        self.trees.append(self.root)

        if self.root:
            self.tree_id += 1
            self.node_id = 0
            self.root = None
            self.depth = 0
            self.tree_parent_queue = deque()
            self.new_tree = True

            if self.tree_id >= self.n_estimators:
                self._is_done = True

    def calculate_leaf_value(self, g, h, existence_mask=None, lambda_reg=1.0):
        if existence_mask is not None:
            valid_indices = [i for i, mask in enumerate(existence_mask) if mask == 1]
            g_valid = g[valid_indices]
            h_valid = h[valid_indices]
        else:
            g_valid = g
            h_valid = h
        
        numerator = -np.sum(g_valid)
        denominator = np.sum(h_valid) + lambda_reg
        
        if denominator == 0:
            return 0.0

        value = numerator / denominator
        return value

    def _assign_leaf_values(self, node, g, h):
        if node is None:
            return
        
        if node.left is None and node.right is None:
            node.value = self.calculate_leaf_value(g, h, node.existence_mask)
            return
        
        self._assign_leaf_values(node.left, g, h)
        self._assign_leaf_values(node.right, g, h)

    def predict(self, X):
        if not self.trees:
            raise ValueError("Model has not been trained yet. No trees available for prediction.")
        
        if self.task_type == "regression":
            predictions = np.zeros(len(X))
        elif self.task_type == "binary":
            predictions = np.zeros(len(X))
        elif self.task_type == "multiclass":
            n_classes = len(np.unique(self.y))
            predictions = np.zeros((len(X), n_classes))
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        for tree in self.trees:
            for i in range(1, len(X)):
                sample = X.iloc[i:i+1]
                tree_pred = self._predict_single_tree(tree, sample)
                
                if self.task_type == "regression":
                    predictions[i] += self.learning_rate * tree_pred
                elif self.task_type == "binary":
                    predictions[i] += self.learning_rate * tree_pred
                elif self.task_type == "multiclass":
                    predictions[i, :] += self.learning_rate * tree_pred
        
        if self.task_type == "regression":
            return predictions
        elif self.task_type == "binary":
            return (sigmoid(predictions) > 0.5).astype(int)
        elif self.task_type == "multiclass":
            return np.argmax(softmax(predictions), axis=1)

    def _predict_single_tree(self, node, X):
        if node.left is None and node.right is None:
            return node.value
        
        feature_value = X[node.feature_name].iloc[0]
        
        if feature_value < node.threshold:
            return self._predict_single_tree(node.left, X)
        else:
            return self._predict_single_tree(node.right, X)