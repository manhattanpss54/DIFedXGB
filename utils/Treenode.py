class Treenode:
    """
    A tree node class for XGBoost-style decision trees in federated learning.
    Represents both internal split nodes and leaf nodes in the decision tree structure.
    """
    
    def __init__(self, 
                 tree_id=None, 
                 node_id=None,
                 feature_idx=None,
                 feature_name=None,
                 threshold=None,
                 value=None,
                 left=None,
                 right=None,
                 existence_mask=None,
                 depth=None,
                 parent=None):
        """
        Initialize a tree node with the following parameters:
        
        Args:
            tree_id: Identifier for the tree this node belongs to
            node_id: Unique identifier for this node within the tree
            feature_idx: Index of the splitting feature (deprecated)
            feature_name: Name of the splitting feature
            threshold: Threshold value for splitting
            value: Prediction value (for leaf nodes)
            left: Left child node
            right: Right child node
            existence_mask: Mask indicating active samples
            depth: Depth of this node in the tree
            parent: Parent node reference
        """
        self.tree_id = tree_id
        self.node_id = node_id
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.existence_mask = existence_mask
        self.depth = depth
        self.parent = parent
        self.left_subtree = False