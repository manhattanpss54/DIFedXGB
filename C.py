from typing import Any
import numpy as np

class C:
    
    def __init__(self):
        pass

    def set_gh(self, g, h):
        self.gradients = np.array(g)
        self.hessians = np.array(h)

    def aggregate_inf(self, a_res, b_res, a_existence_mask, b_existence_mask):
        existence_mask = self.combine_masks(a_existence_mask, b_existence_mask)
        max_gain = -float('inf')
        best_split = None

        for res in a_res + b_res:
            feature_name, thr, mask = res
            mask_list = mask.tolist() if hasattr(mask, 'tolist') else list(mask)

            if len(mask_list) != len(self.gradients):
                raise ValueError(f"Mask length mismatch")

            left_mask = [1 if a == 1 and m == 0 else 0 for a, m in zip(existence_mask, mask_list)]
            right_mask = [1 if a == 1 and m == 1 else 0 for a, m in zip(existence_mask, mask_list)]

            left_bool_mask = np.array(left_mask, dtype=bool)
            right_bool_mask = np.array(right_mask, dtype=bool)

            left_g = self.gradients[left_bool_mask]
            left_h = self.hessians[left_bool_mask]
            right_g = self.gradients[right_bool_mask]
            right_h = self.hessians[right_bool_mask]

            if len(left_g) == 0 or len(right_g) == 0:
                continue

            gain = self._calculate_gain(left_g, left_h, right_g, right_h)

            if gain > max_gain:
                max_gain = gain
                best_split = (feature_name, thr)
        
        return best_split

    def _calculate_gain(self, left_g, left_h, right_g, right_h):
        G_L = np.sum(left_g)
        H_L = np.sum(left_h)
        G_R = np.sum(right_g)
        H_R = np.sum(right_h)
        
        gain = (G_L**2 / (H_L + 1) + G_R**2 / (H_R + 1) - 
               (G_L + G_R)**2 / (H_L + H_R + 1)) / 2
        
        return gain
    
    def combine_masks(self, a, b):
        return [1 if a == 1 and b == 1 else 0 for a, b in zip(a, b)]