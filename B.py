from collections import deque
import pandas as pd
import os
from utils.Treenode import Treenode
from utils.infer_task_type import infer_task_type
from utils.create_equal_width_bins import create_equal_width_bins
from utils.activation import sigmoid, softmax
import numpy as np
from sklearn.preprocessing import LabelEncoder

class B:
    _is_done = False

    def __init__(self, 
                 global_features=None, 
                 local_features=None,
                 label=None, 
                 max_depth=3, 
                 n_estimators=10,
                 learning_rate=0.1):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'dataset', 'processed', 'dataset_B.csv')
        self.dataset = pd.read_csv(path)
        self.X = self.dataset

        self.tree_id = 0
        self.node_id = 0
        self.root = None
        self.trees = []
        self.depth = 0
        self.max_depth = max_depth
        self.tree_parent_queue = deque()
        self.n_estimators = n_estimators
        self.global_features = global_features
        self.local_features = local_features
        self.max_gain = 0
        self.best_feature = None
        self.best_threshold = None

    def upload_inf(self):
        existence_mask = self.get_existence_mask()
        feature_name_thr_mask = self.get_feature_name_thr_mask()
        return existence_mask, feature_name_thr_mask
    
    def create_node(self, feature_name, thr):
        existence_mask = self.get_existence_mask()

        if feature_name in self.local_features: 
            thr_mask = np.where(self.X[feature_name] < thr, 0, 1)
            existence_mask = [-1 if e == 0 else t for e, t in zip(existence_mask, thr_mask)]
        else:
            existence_mask = existence_mask

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
            self.new_tree = True

        self.node_id += 1
        self.tree_parent_queue.append(new_node)
        
    def _finalize_tree(self):
        if self.root:
            self.trees.append(self.root)
            self.tree_id += 1
            self.node_id = 0
            self.root = None
            self.depth = 0
            self.tree_parent_queue = deque()
            self.new_tree = True

            if self.tree_id >= self.n_estimators:
                self._is_done = True

    def get_existence_mask(self):
        if self.tree_parent_queue and len(self.tree_parent_queue) > 0:
            parent_node = self.tree_parent_queue[0]
            existence_mask = parent_node.existence_mask
            if parent_node.left is None:
                existence_mask = [1 if x == 0 else 0 if x == 1 else x for x in existence_mask]
            else:
                existence_mask = [0 if x == 0 else 1 if x == 1 else x for x in existence_mask]
        else:
            existence_mask = np.full_like(self.X.iloc[:, 0], 1) 

        return existence_mask

    def get_feature_name_thr_mask(self):
        feature_name_thr_mask = []
        for feature_name in self.local_features:
            thrs = create_equal_width_bins(feature_data=self.X[feature_name])
            for thr in thrs:
                thr_mask = np.where(self.X[feature_name] < thr, 0, 1)
                feature_name_thr_mask.append([feature_name, thr, thr_mask])
        return feature_name_thr_mask