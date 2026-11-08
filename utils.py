import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from math import sqrt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    if isinstance(index, list):
        index = torch.tensor(index)
    mask[index] = 1
    return mask


def take_rest(x, y):
    """从x中移除y中的元素"""
    x.sort()
    y.sort()
    res = []
    j, jmax = 0, len(y)
    for i in range(0, len(x)):
        flag = False
        while j < jmax and y[j] <= x[i]:
            if y[j] == x[i]:
                flag = True
            j += 1
        if not flag:
            res.append(x[i])
    return res


def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    min_class_size = min([len(np.where(data.y.cpu() == c)[0]) for c in range(num_classes)])
    adjusted_percls_trn = min(percls_trn, int(0.6 * min_class_size))

    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        train_idx.extend(rnd_state.choice(class_idx, adjusted_percls_trn, replace=False))
    actual_train_size = len(train_idx)
    total_nodes = len(data.y)

    if actual_train_size < int(0.6 * total_nodes):
        remaining_nodes = total_nodes - actual_train_size
        val_lb = int(0.2 * total_nodes)
        val_lb = min(val_lb, remaining_nodes - 1)

    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
        test_idx = [i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
        test_idx = take_rest(rest_index, val_idx)
    
    print(f"Total: {data.y.shape[0]}, Train: {len(train_idx)} ({len(train_idx)/data.y.shape[0]*100:.1f}%), "
          f"Val: {len(val_idx)} ({len(val_idx)/data.y.shape[0]*100:.1f}%), "
          f"Test: {len(test_idx)} ({len(test_idx)/data.y.shape[0]*100:.1f}%)")

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    
    return data

def random_splits_miss(data, num_classes, percls_trn, val_lb, seed=42):
    return random_splits(data, num_classes, percls_trn, val_lb, seed)


def random_splits_citation(data, num_classes, percls_trn, val_lb, seed=42):
    return random_splits(data, num_classes, percls_trn, val_lb, seed)


def fixed_splits(data):
    return data


def batch_generator(data, batch_size):
    pass


class DataSplit:
    pass
