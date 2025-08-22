import numpy as np

def infer_task_type(y):
    y = np.array(y)
    
    # 情况1: y是二维数组（可能是One-Hot编码的多分类）
    if y.ndim == 2 and y.shape[1] > 1:
        return "multiclass"
    
    # 情况2: y是整数且唯一值数量 > 2 → 多分类
    unique_values = np.unique(y)
    if len(unique_values) > 2 and np.issubdtype(y.dtype, np.integer):
        return "multiclass"
    
    # 情况3: y是整数且唯一值数量 = 2 → 二分类
    elif len(unique_values) == 2 and np.issubdtype(y.dtype, np.integer):
        return "binary"
    
    # 情况4: 其他情况（浮点数或连续值）→ 回归
    else:
        return "regression"