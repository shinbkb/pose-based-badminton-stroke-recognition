"""COCO and H36M skeleton adjacency matrices."""
import numpy as np
import torch


COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def normalize_digraph(matrix):
    degree = matrix.sum(axis=0)

    inverse_degree = np.zeros_like(
        degree,
        dtype=np.float32,
    )

    valid = degree > 0

    inverse_degree[valid] = (
        1.0 / degree[valid]
    )

    return (
        matrix
        @ np.diag(inverse_degree)
    )


def build_coco_adjacency():
    num_joints = 17

    identity = np.eye(
        num_joints,
        dtype=np.float32,
    )

    inward = np.zeros(
        (num_joints, num_joints),
        dtype=np.float32,
    )

    for source, target in COCO_EDGES:
        inward[target, source] = 1.0

    outward = inward.T

    adjacency = np.stack(
        [
            identity,
            normalize_digraph(inward),
            normalize_digraph(outward),
        ]
    )

    return torch.tensor(
        adjacency,
        dtype=torch.float32,
    )


H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def build_h36m_adjacency():
    num_joints = 17

    identity = np.eye(
        num_joints,
        dtype=np.float32,
    )

    inward = np.zeros(
        (num_joints, num_joints),
        dtype=np.float32,
    )

    for source, target in H36M_EDGES:
        inward[target, source] = 1.0

    outward = inward.T

    adjacency = np.stack(
        [
            identity,
            normalize_digraph(inward),
            normalize_digraph(outward),
        ]
    )

    return torch.tensor(
        adjacency,
        dtype=torch.float32,
    )
