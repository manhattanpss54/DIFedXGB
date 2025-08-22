import numpy as np

def infer_task_type(y):
    y = np.array(y)
    
    if y.ndim == 2 and y.shape[1] > 1:
        return "multiclass"
    
    unique_values = np.unique(y)
    if len(unique_values) > 2 and np.issubdtype(y.dtype, np.integer):
        return "multiclass"
    elif len(unique_values) == 2 and np.issubdtype(y.dtype, np.integer):
        return "binary"
    else:
        return "regression"