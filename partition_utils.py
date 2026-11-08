
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_scipy_sparse_matrix
import math


def spectral_partition(edge_index, num_nodes, num_partitions, device):
    # Convert to scipy sparse matrix for eigendecomposition
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    L_scipy = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    # Compute degree matrix
    deg = np.array(L_scipy.sum(axis=1)).flatten()
    deg[deg == 0] = 1  # Avoid division by zero
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))

    # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    A = L_scipy
    L_norm = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute smallest k eigenvectors (spectral embedding)
    # Optimization: Use shift-invert to convert SM to LM (3-4x faster)
    k = min(num_partitions + 1, num_nodes - 2)
    try:
        offset = 2.0
        T = offset * sp.eye(num_nodes) - L_norm
        eigenvalues, eigenvectors = eigsh(T, k=k, which='LM', tol=1e-3)
        # Convert back to L's eigenvalues (use .copy() to avoid negative strides)
        eigenvalues = (offset - eigenvalues)[::-1].copy()
        eigenvectors = np.ascontiguousarray(eigenvectors[:, ::-1])
    except:
        print("false")
        # Fallback: use random partition if eigendecomposition fails
        partition_matrix = torch.randn(num_nodes, num_partitions, device=device)
        partition_matrix = F.softmax(partition_matrix, dim=1)
        return partition_matrix

    spectral_coords = eigenvectors[:, 1:num_partitions + 1]  # (num_nodes, num_partitions)
    spectral_coords_tensor = torch.from_numpy(spectral_coords).float().to(device)

    fiedler_vector = spectral_coords[:, 0]
    sorted_indices = np.argsort(fiedler_vector)

    partition_centers = []
    for i in range(num_partitions):
        start_idx = int(i * num_nodes / num_partitions)
        end_idx = int((i + 1) * num_nodes / num_partitions)

        if i == num_partitions - 1:
            end_idx = num_nodes

        group_indices = sorted_indices[start_idx:end_idx]
        group_coords = spectral_coords[group_indices]

        if len(group_coords) == 0:
            if len(partition_centers) > 0:
                partition_centers.append(partition_centers[-1])
            else:
                partition_centers.append(spectral_coords[0])
        elif len(group_coords) == 1:
            partition_centers.append(group_coords[0])
        else:
            centroid = group_coords.mean(axis=0)
            distances = np.linalg.norm(group_coords - centroid, axis=1)
            center_idx_in_group = distances.argmin()
            partition_centers.append(group_coords[center_idx_in_group])

    partition_centers = torch.tensor(np.array(partition_centers),
                                     dtype=torch.float32, device=device)

    # Compute soft assignment based on distance to centers
    # partition_matrix[i, j] = exp(-||x_i - c_j||^2 / (2 * sigma^2))
    distances = torch.cdist(spectral_coords_tensor, partition_centers, p=2)  # (num_nodes, num_partitions)
    sigma = distances.std()
    if sigma < 1e-6:
        sigma = 1.0
    partition_matrix = torch.exp(-distances ** 2 / (2 * sigma ** 2))

    # Normalize to make each row sum to 1 (soft assignment)
    partition_matrix = F.normalize(partition_matrix, p=1, dim=1)

    return partition_matrix




def frequency_energy_partition(node_feat, edge_index, num_nodes, num_partitions, device):
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    A_scipy = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    deg = np.array(A_scipy.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    L_norm = sp.eye(num_nodes) - D_inv_sqrt @ A_scipy @ D_inv_sqrt

    k_freq = min(50, num_nodes - 2)
    k_per_band = max(k_freq // 2, 1)

    try:
        offset = 2.0
        T = offset * sp.eye(num_nodes) - L_norm
        evals_low_T, evecs_low = eigsh(T, k=k_per_band, which='LM', tol=1e-3)
        evals_low = (offset - evals_low_T)[::-1].copy()
        evecs_low = np.ascontiguousarray(evecs_low[:, ::-1])
        evals_high_raw, evecs_high_raw = eigsh(L_norm, k=k_per_band, which='LM', tol=1e-3)

        sort_high = np.argsort(evals_high_raw)[::-1]
        evals_high = np.ascontiguousarray(evals_high_raw[sort_high])
        evecs_high = np.ascontiguousarray(evecs_high_raw[:, sort_high])

        eigenvalues = np.concatenate([evals_low, evals_high])
        eigenvectors = np.concatenate([evecs_low, evecs_high], axis=1)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        if not np.isfinite(eigenvectors).all():
            raise ValueError("Eigenvectors contain NaN or Inf")

    except Exception as e:
        print(f"Eigendecomposition failed: {e}")
        partition_matrix = torch.randn(num_nodes, num_partitions, device=device)
        return F.softmax(partition_matrix, dim=1)
    U = torch.from_numpy(eigenvectors).float().to(device)  # (N, total_k)
    node_feat_normalized = F.normalize(node_feat, p=2, dim=1)  # (N, F)
    freq_coeffs = torch.mm(U.T, node_feat_normalized)  # (total_k, F)
    freq_energy = torch.zeros(num_nodes, num_partitions, device=device)

    low_parts = num_partitions // 2
    high_parts = num_partitions - low_parts

    for p in range(low_parts):
        start = p * k_per_band // low_parts
        end = (p + 1) * k_per_band // low_parts if p < low_parts - 1 else k_per_band
        U_band = U[:, start:end]
        coeffs_band = freq_coeffs[start:end, :]
        feat_band_reconstructed = torch.mm(U_band, coeffs_band)
        freq_energy[:, p] = (feat_band_reconstructed ** 2).sum(dim=1)

    total_k = U.shape[1]
    for p in range(high_parts):
        start = k_per_band + p * k_per_band // high_parts
        end = k_per_band + (p + 1) * k_per_band // high_parts if p < high_parts - 1 else total_k
        U_band = U[:, start:end]
        coeffs_band = freq_coeffs[start:end, :]
        feat_band_reconstructed = torch.mm(U_band, coeffs_band)
        freq_energy[:, low_parts + p] = (feat_band_reconstructed ** 2).sum(dim=1)

    partition_matrix = F.normalize(freq_energy + 1e-8, p=1, dim=1)

    return partition_matrix

def partition_index(dataset, num_partitions, device):
    feature = dataset.graph['node_feat'].to(device)
    feature = F.normalize(feature, p=2, dim=1)
    edges = dataset.graph['edge_index'].to(device)
    num_nodes = dataset.graph['num_nodes']
    
    print(f'--Computing structure-aware partition via spectral embedding...')
    partition_struct = spectral_partition(edges, num_nodes, num_partitions, device)

    print(f'--Computing frequency-aware partition via spectral energy...')
    partition_freq = frequency_energy_partition(feature, edges, num_nodes, num_partitions, device)

    return feature, edges, partition_struct, partition_freq
