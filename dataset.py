"""
Dataset for Badminton Stroke Recognition with Data Augmentation.

Joint files: (T, M, V, C) — time, persons, joints, coords (x,y)
Model input: (C, T, V, M) after transpose
"""

import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
JOINT_DIR = 'processed/joints_npy_yolo26s_pose'
SCORE_DIR = 'processed/scores_npy_yolo26s_pose'
CLASS_NAMES = [
    '00_Short Serve', '01_Cross Court Flight', '02_Lift', '03_Tap Smash',
    '04_Block', '05_Drop Shot', '06_Push Shot', '07_Transitional Slice',
    '08_Cut', '09_Rush Shot', '10_Defensive Clear', '11_Defensive Drive',
    '12_Clear', '13_Long Serve', '14_Smash', '15_Flat Shot',
    '16_Rear Court Flat Drive', '17_Short Flat Shot',
]
CLASS2IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

TARGET_T = 64      # Fixed temporal length
NUM_JOINTS = 17
NUM_PERSONS = 2
IN_CHANNELS = 2    # x, y


# ─────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────
class TemporalCrop:
    """Randomly crop to target length, or pad if shorter."""
    def __init__(self, target_len: int = TARGET_T):
        self.target_len = target_len

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T >= self.target_len:
            start = random.randint(0, T - self.target_len)
            return seq[start: start + self.target_len]
        else:
            # Pad by repeating last frame
            pad = np.repeat(seq[-1:], self.target_len - T, axis=0)
            return np.concatenate([seq, pad], axis=0)


class CenterCrop:
    """Deterministic center crop (for val/test)."""
    def __init__(self, target_len: int = TARGET_T):
        self.target_len = target_len

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T >= self.target_len:
            start = (T - self.target_len) // 2
            return seq[start: start + self.target_len]
        else:
            pad = np.repeat(seq[-1:], self.target_len - T, axis=0)
            return np.concatenate([seq, pad], axis=0)


class JointNoise:
    """Add small Gaussian noise to joint coordinates."""
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*seq.shape).astype(np.float32) * self.sigma
        return seq + noise


class HorizontalFlip:
    """Mirror left-right by negating x-coordinate (coord index 0)."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        # seq: (T, M, V, C)  C=0 is x
        if random.random() < self.p:
            seq = seq.copy()
            seq[..., 0] = 1.0 - seq[..., 0]
            # Also swap left/right joints (COCO symmetric pairs)
            LEFT  = [1, 3, 5, 7, 9, 11, 13, 15]
            RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]
            seq_flip = seq.copy()
            seq_flip[:, :, LEFT, :] = seq[:, :, RIGHT, :]
            seq_flip[:, :, RIGHT, :] = seq[:, :, LEFT, :]
            return seq_flip
        return seq


class RandomDropFrames:
    """Randomly zero-out a fraction of frames."""
    def __init__(self, drop_ratio: float = 0.1):
        self.drop_ratio = drop_ratio

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        n_drop = int(T * self.drop_ratio)
        if n_drop > 0:
            seq = seq.copy()
            drop_idx = np.random.choice(T, size=n_drop, replace=False)
            seq[drop_idx] = 0.0
        return seq


class NormalizeCoords:
    """Normalize coordinates to zero-mean, unit-variance per sequence."""
    def __call__(self, seq: np.ndarray) -> np.ndarray:
        # seq: (T, M, V, C)
        mean = seq.mean(axis=(0, 2), keepdims=True)   # per person, per coord
        std  = seq.std(axis=(0, 2), keepdims=True) + 1e-6
        return (seq - mean) / std


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class BadmintonDataset(Dataset):
    """
    Loads skeleton joint sequences for badminton stroke classification.

    Args:
        split: 'train', 'val', or 'test'
        augment: whether to apply training augmentations
        target_len: fixed temporal length after crop/pad
        fix_label_noise: remove files with mismatched class suffix
    """

    def __init__(
        self,
        split: str = 'train',
        augment: bool = True,
        target_len: int = TARGET_T,
        fix_label_noise: bool = True,
    ):
        self.split = split
        self.augment = augment and (split == 'train')
        self.target_len = target_len

        # Build transforms
        if self.augment:
            self.crop = TemporalCrop(target_len)
            self.aug_transforms = [
                JointNoise(sigma=0.01),
                HorizontalFlip(p=0.5),
                RandomDropFrames(drop_ratio=0.1),
            ]
        else:
            self.crop = CenterCrop(target_len)
            self.aug_transforms = []

        self.norm = NormalizeCoords()

        # Collect samples
        self.samples = []   # list of (joint_path, score_path, label_idx)
        joint_split_dir = os.path.join(JOINT_DIR, split)
        score_split_dir = os.path.join(SCORE_DIR, split)

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(joint_split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label_idx = CLASS2IDX[class_name]
            class_suffix = class_name.split('_')[0]  # e.g. '07'

            for fname in os.listdir(class_dir):
                if not fname.endswith('_joints.npy'):
                    continue

                # Fix label noise: skip files whose suffix doesn't match class
                if fix_label_noise:
                    # Filename pattern: ..._<classidx>_joints.npy
                    parts = fname.replace('_joints.npy', '').split('_')
                    file_class_suffix = parts[-1]  # last part before _joints
                    if file_class_suffix != class_suffix:
                        continue

                joint_path = os.path.join(class_dir, fname)
                score_fname = fname.replace('_joints.npy', '_scores.npy')
                score_path = os.path.join(score_split_dir, class_name, score_fname)
                if not os.path.exists(score_path):
                    score_path = None

                self.samples.append((joint_path, score_path, label_idx))

        print(f'[Dataset] {split}: {len(self.samples)} samples, augment={self.augment}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        joint_path, score_path, label_idx = self.samples[idx]

        # Load joints: (T, M, V, C) — time, persons, joints, coords
        joints = np.load(joint_path).astype(np.float32)

        # Handle shape variations
        if joints.ndim == 3:
            # (T, V, C) — single person → expand person dim
            joints = joints[:, np.newaxis, :, :]  # T,1,V,C

        # Ensure shape (T, M, V, C) with M=2
        T, M, V, C = joints.shape
        if M < NUM_PERSONS:
            pad = np.zeros((T, NUM_PERSONS - M, V, C), dtype=np.float32)
            joints = np.concatenate([joints, pad], axis=1)
        joints = joints[:, :NUM_PERSONS, :NUM_JOINTS, :IN_CHANNELS]

        # Confidence-weighted masking (set low-confidence joints to 0)
        if score_path is not None and os.path.exists(score_path):
            scores = np.load(score_path).astype(np.float32)  # (T, M, V) or similar
            if scores.ndim == 2:
                scores = scores[:, np.newaxis, :]  # T,1,V
            scores = scores[:, :NUM_PERSONS, :NUM_JOINTS]
            # Zero out joints with confidence < 0.3
            mask = (scores < 0.3)[..., np.newaxis]  # T,M,V,1
            joints = np.where(mask, 0.0, joints)

        # Apply augmentations (before crop so they see full seq)
        if self.augment:
            for t in self.aug_transforms:
                joints = t(joints)

        # Temporal crop/pad to fixed length
        joints = self.crop(joints)  # (target_len, M, V, C)

        # Normalize coordinates
        joints = self.norm(joints)

        # Transpose to model input: (C, T, V, M)
        joints = joints.transpose(3, 0, 2, 1)  # C, T, V, M

        return torch.from_numpy(joints), label_idx

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for _, _, label_idx in self.samples:
            counts[label_idx] += 1
        # Classes with 0 samples get weight 0
        weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
        # Normalize so mean weight = 1
        weights = weights / (weights[counts > 0].mean() + 1e-6)
        return torch.from_numpy(weights).float()


def build_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    target_len: int = TARGET_T,
):
    train_ds = BadmintonDataset('train', augment=True, target_len=target_len)
    val_ds   = BadmintonDataset('val',   augment=False, target_len=target_len)
    test_ds  = BadmintonDataset('test',  augment=False, target_len=target_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds.get_class_weights()
