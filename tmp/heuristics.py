from typing_extensions import final
from collections import defaultdict
import math

from numpy.lib.twodim_base import mask_indices
import torch
import numpy as np
from itertools import groupby

from torch import nn
from torch.nn import functional as F


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio, intercept

ratio = torch.tensor([[[0.8017, 0.7030, 0.6828, 0.1961, 0.3994, 0.3634, 0.7698, 0.8867,
          0.9313, 0.5039]],
        [[0.1914, 0.7184, 0.8463, 0.8017, 0.1001, 0.2627, 0.8231, 0.8939,
          0.3224, 0.4989]]])
lower = torch.tensor([[ 0.0733, -0.4172,  0.4147,  0.2980, -0.0005,  0.2553, -0.2080, -0.1256,
          0.4952, -0.1191],
        [-0.2387,  0.4561, -0.2835, -0.4698,  0.2590,  0.0367, -0.1935,  0.3723,
         -0.0880, -0.2540]])
upper = lower + 0.2
this_layer_mask = torch.logical_and(lower < 0, upper > 0)
ratio_temp_0, ratio_temp_1 = compute_ratio(lower, upper)
intercept_temp = torch.clamp(ratio, max=0) 
intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
b_temp = torch.tensor([0.1092, 0.3603, 0.0835, 0.1580, 0.2360, 0.6852, 0.2818, 0.0740, 0.6898,
        0.1128])
b_temp = b_temp * ratio#rati0 -> A(v)
# Estimated bounds of the two sides of the bounds.
ratio_temp_0 = ratio_temp_0.unsqueeze(1)
bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
bias_candidate_2 = b_temp * ratio_temp_0
bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)  # max for babsr by default
score_candidate = bias_candidate + intercept_candidate
print(score_candidate)
print(score_candidate.shape)
score_candidate = score_candidate.abs() * this_layer_mask.unsqueeze(1)
score_candidate = score_candidate.mean(1)
print(score_candidate.shape)
print(score_candidate)
                                                   