from numbers import Number

def create_equal_width_bins(feature_data, num_bins=10, is_numeric=None):
    if len(feature_data) == 0:
        return []

    if is_numeric is None:
        is_numeric = all(isinstance(x, Number) for x in feature_data)
    
    if is_numeric:
        min_val = min(feature_data)
        max_val = max(feature_data)
        
        if min_val == max_val:
            return []
        
        step = (max_val - min_val) / num_bins
        return [min_val + i * step for i in range(1, num_bins)]
    else:
        unique_values = sorted(list(set(feature_data)))
        
        if len(unique_values) <= num_bins:
            return unique_values 
        
        bin_size = len(unique_values) / num_bins
        
        split_points = []
        for i in range(1, num_bins):
            split_index = int(i * bin_size)
            if split_index < len(unique_values):
                split_points.append(unique_values[split_index])
        
        return split_points