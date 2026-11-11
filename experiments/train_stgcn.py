"""Exported from notebooks/stgcn_experiments.ipynb.

Regenerate with: python tools/export_notebooks.py
Run --help for usage. Importing this script does not start the workflow.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Use the Agg plotting backend when running without a desktop.",
    )
    args = parser.parse_args()
    from IPython.display import display

    if args.non_interactive:
        os.environ["MPLBACKEND"] = "Agg"
    os.chdir(_ROOT)

    # %% Cell 1
    from pathlib import Path
    import copy
    import json
    import random
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm


    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # %% Cell 2
    from badminton_stroke.paths import PROJECT_ROOT

    JOINTS_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "joints_npy_yolo26s_pose"
    )

    OUTPUT_ROOT = (
        EXPERIMENTS_DIR / "stgcn_v2"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Dữ liệu
    TARGET_FRAMES = 64
    CONFIDENCE_THRESHOLD = 0.20

    # DataLoader
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Training
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 12

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.05

    # Augmentation
    HORIZONTAL_FLIP_PROBABILITY = 0.5
    JOINT_NOISE_SIGMA = 0.01
    FRAME_DROP_PROBABILITY = 0.05


    assert JOINTS_ROOT.exists(), (
        f"Không tìm thấy: {JOINTS_ROOT}"
    )

    print("Joints root:", JOINTS_ROOT)
    print("Output root:", OUTPUT_ROOT)

    # %% Cell 3
    def find_score_path(joint_path):
        candidates = [
            joint_path.with_name(
                joint_path.name.replace(
                    "_joints.npy",
                    "_scores.npy",
                )
            ),
            joint_path.with_name(
                joint_path.name.replace(
                    "joints.npy",
                    "scores.npy",
                )
            ),
        ]

        for candidate in candidates:
            if (
                candidate.exists()
                and candidate != joint_path
            ):
                return candidate

        return None


    def build_index(split):
        split_root = JOINTS_ROOT / split

        assert split_root.exists(), (
            f"Không tìm thấy split: {split_root}"
        )

        records = []

        joint_files = sorted(
            split_root.rglob("*_joints.npy")
        )

        for joint_path in joint_files:
            relative_path = (
                joint_path.relative_to(split_root)
            )

            if len(relative_path.parts) < 2:
                raise RuntimeError(
                    f"{joint_path} không nằm trong "
                    "thư mục class."
                )

            label_name = relative_path.parts[0]
            score_path = find_score_path(joint_path)

            records.append(
                {
                    "split": split,
                    "label_name": label_name,
                    "joint_path": str(joint_path),
                    "score_path": (
                        str(score_path)
                        if score_path is not None
                        else ""
                    ),
                }
            )

        frame = pd.DataFrame(records)

        assert len(frame) > 0, (
            f"Không tìm thấy *_joints.npy "
            f"trong {split_root}"
        )

        return frame


    train_df = build_index("train")
    val_df = build_index("val")
    test_df = build_index("test")

    class_names = sorted(
        train_df["label_name"]
        .unique()
        .tolist()
    )

    class_to_id = {
        class_name: class_id
        for class_id, class_name
        in enumerate(class_names)
    }

    for split_name, frame in {
        "val": val_df,
        "test": test_df,
    }.items():
        unknown_classes = (
            set(frame["label_name"])
            - set(class_names)
        )

        assert not unknown_classes, (
            f"{split_name} có lớp không xuất hiện "
            f"trong train: {unknown_classes}"
        )

    print("Số lớp:", len(class_names))
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("Các lớp:")

    for class_id, class_name in enumerate(class_names):
        print(class_id, class_name)

    display(train_df.head())

    # %% Cell 4
    all_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    audit_records = []

    for row in tqdm(
        all_df.itertuples(index=False),
        total=len(all_df),
        desc="Audit dữ liệu",
    ):
        try:
            joints = np.load(
                row.joint_path,
                mmap_mode="r",
            )

            valid_shape = (
                joints.ndim == 4
                and joints.shape[1:] == (2, 17, 2)
            )

            audit_records.append(
                {
                    "split": row.split,
                    "label_name": row.label_name,
                    "num_frames": (
                        joints.shape[0]
                        if joints.ndim > 0
                        else 0
                    ),
                    "shape": tuple(joints.shape),
                    "valid": valid_shape,
                    "joint_path": row.joint_path,
                    "error": "",
                }
            )

        except Exception as error:
            audit_records.append(
                {
                    "split": row.split,
                    "label_name": row.label_name,
                    "num_frames": 0,
                    "shape": None,
                    "valid": False,
                    "joint_path": row.joint_path,
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

    audit_df = pd.DataFrame(audit_records)

    print(
        "File hợp lệ:",
        int(audit_df["valid"].sum()),
        "/",
        len(audit_df),
    )

    display(
        audit_df.loc[
            audit_df["valid"],
            "num_frames",
        ].describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    invalid_df = audit_df[
        ~audit_df["valid"]
    ]

    if len(invalid_df):
        display(invalid_df.head(20))

    # %% Cell 5
    class_distribution = (
        all_df
        .groupby(["label_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(class_names)
    )

    display(class_distribution)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 5),
    )

    class_distribution.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
    )

    axes[0].set_title("Phân bố lớp")
    axes[0].set_xlabel("Kỹ thuật")
    axes[0].set_ylabel("Số video")
    axes[0].tick_params(
        axis="x",
        rotation=70,
    )

    sns.histplot(
        data=audit_df[audit_df["valid"]],
        x="num_frames",
        hue="split",
        bins=40,
        ax=axes[1],
    )

    axes[1].axvline(
        TARGET_FRAMES,
        color="red",
        linestyle="--",
        label=f"TARGET_FRAMES={TARGET_FRAMES}",
    )

    axes[1].set_title("Phân bố số frame")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # %% Cell 6
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    COCO_LEFT_RIGHT_PAIRS = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
    ]


    def load_scores(score_path, joints):
        if score_path:
            scores = np.load(
                score_path
            ).astype(np.float32)

            if scores.shape == joints.shape[:-1]:
                return scores

        # Fallback nếu không có score file.
        return np.any(
            joints != 0,
            axis=-1,
        ).astype(np.float32)


    def normalize_skeleton(
        joints,
        scores,
        confidence_threshold,
    ):
        joints = joints.astype(
            np.float32,
            copy=True,
        )

        scores = scores.astype(
            np.float32,
            copy=True,
        )

        valid = (
            scores >= confidence_threshold
        )

        num_frames, num_people, _, _ = (
            joints.shape
        )

        normalized = np.zeros_like(
            joints,
            dtype=np.float32,
        )

        for person_index in range(num_people):
            person_xy = joints[
                :, person_index
            ]

            person_valid = valid[
                :, person_index
            ]

            centers = np.zeros(
                (num_frames, 2),
                dtype=np.float32,
            )

            torso_lengths = []

            for frame_index in range(num_frames):
                hip_valid = person_valid[
                    frame_index,
                    [LEFT_HIP, RIGHT_HIP],
                ]

                if hip_valid.all():
                    centers[frame_index] = (
                        person_xy[
                            frame_index,
                            [LEFT_HIP, RIGHT_HIP],
                        ].mean(axis=0)
                    )

                elif person_valid[
                    frame_index
                ].any():
                    centers[frame_index] = (
                        person_xy[
                            frame_index,
                            person_valid[frame_index],
                        ].mean(axis=0)
                    )

                torso_valid = person_valid[
                    frame_index,
                    [
                        LEFT_SHOULDER,
                        RIGHT_SHOULDER,
                        LEFT_HIP,
                        RIGHT_HIP,
                    ],
                ]

                if torso_valid.all():
                    shoulder_center = person_xy[
                        frame_index,
                        [
                            LEFT_SHOULDER,
                            RIGHT_SHOULDER,
                        ],
                    ].mean(axis=0)

                    hip_center = person_xy[
                        frame_index,
                        [
                            LEFT_HIP,
                            RIGHT_HIP,
                        ],
                    ].mean(axis=0)

                    torso_lengths.append(
                        np.linalg.norm(
                            shoulder_center
                            - hip_center
                        )
                    )

            centered = (
                person_xy
                - centers[:, None, :]
            )

            if torso_lengths:
                scale = float(
                    np.median(torso_lengths)
                )

            else:
                visible_points = centered[
                    person_valid
                ]

                if len(visible_points):
                    scale = float(
                        np.max(
                            np.ptp(
                                visible_points,
                                axis=0,
                            )
                        )
                    )
                else:
                    scale = 1.0

            scale = max(scale, 1e-6)

            centered /= scale
            centered[~person_valid] = 0.0

            normalized[:, person_index] = (
                centered
            )

        scores[~valid] = 0.0

        return normalized, scores

    # %% Cell 7
    def crop_or_pad(
        joints,
        scores,
        target_frames,
        training,
    ):
        """
        Giữ nguyên frame rate.

        Video dài:
            random crop khi train,
            center crop khi val/test.

        Video ngắn:
            padding về target_frames.
        """

        num_frames = len(joints)

        if num_frames >= target_frames:
            max_start = (
                num_frames - target_frames
            )

            if training and max_start > 0:
                start = random.randint(
                    0,
                    max_start,
                )
            else:
                start = max_start // 2

            end = start + target_frames

            return (
                joints[start:end],
                scores[start:end],
            )

        output_joints = np.zeros(
            (
                target_frames,
                *joints.shape[1:],
            ),
            dtype=np.float32,
        )

        output_scores = np.zeros(
            (
                target_frames,
                *scores.shape[1:],
            ),
            dtype=np.float32,
        )

        empty_frames = (
            target_frames - num_frames
        )

        if training and empty_frames > 0:
            start = random.randint(
                0,
                empty_frames,
            )
        else:
            start = empty_frames // 2

        end = start + num_frames

        output_joints[start:end] = joints
        output_scores[start:end] = scores

        return output_joints, output_scores


    def horizontal_flip(joints, scores):
        joints = joints.copy()
        scores = scores.copy()

        # Đảo trục x.
        joints[..., 0] *= -1

        # Đổi joint trái/phải.
        for left, right in (
            COCO_LEFT_RIGHT_PAIRS
        ):
            left_joint = joints[
                :, :, left
            ].copy()

            joints[:, :, left] = (
                joints[:, :, right]
            )

            joints[:, :, right] = (
                left_joint
            )

            left_score = scores[
                :, :, left
            ].copy()

            scores[:, :, left] = (
                scores[:, :, right]
            )

            scores[:, :, right] = (
                left_score
            )

        return joints, scores


    def add_joint_noise(
        joints,
        scores,
        sigma,
    ):
        joints = joints.copy()

        visible = scores > 0

        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=joints.shape,
        ).astype(np.float32)

        joints[visible] += noise[visible]

        return joints


    def random_drop_frames(
        joints,
        scores,
        probability,
    ):
        joints = joints.copy()
        scores = scores.copy()

        for frame_index in range(
            1,
            len(joints),
        ):
            has_pose = np.any(
                scores[frame_index] > 0
            )

            if (
                has_pose
                and random.random() < probability
            ):
                joints[frame_index] = (
                    joints[frame_index - 1]
                )

                scores[frame_index] = (
                    scores[frame_index - 1]
                )

        return joints, scores


    def create_motion(
        joints,
        scores,
    ):
        motion = np.zeros_like(
            joints,
            dtype=np.float32,
        )

        valid_pair = (
            (scores[1:] > 0)
            & (scores[:-1] > 0)
        )

        difference = (
            joints[1:]
            - joints[:-1]
        )

        motion[1:] = np.where(
            valid_pair[..., None],
            difference,
            0.0,
        )

        return motion

    # %% Cell 8
    class BadmintonSTGCNDataset(Dataset):
        def __init__(
            self,
            index_df,
            class_to_id,
            target_frames=64,
            confidence_threshold=0.2,
            training=False,
        ):
            self.index_df = (
                index_df
                .reset_index(drop=True)
                .copy()
            )

            self.class_to_id = class_to_id
            self.target_frames = target_frames
            self.confidence_threshold = (
                confidence_threshold
            )

            self.training = training

        def __len__(self):
            return len(self.index_df)

        def __getitem__(self, index):
            row = self.index_df.iloc[index]

            joints = np.load(
                row["joint_path"]
            ).astype(np.float32)

            if (
                joints.ndim != 4
                or joints.shape[1:]
                != (2, 17, 2)
            ):
                raise ValueError(
                    f"Shape không hợp lệ: "
                    f"{joints.shape}, "
                    f"{row['joint_path']}"
                )

            scores = load_scores(
                row["score_path"],
                joints,
            )

            joints, scores = (
                normalize_skeleton(
                    joints,
                    scores,
                    CONFIDENCE_THRESHOLD,
                )
            )

            joints, scores = crop_or_pad(
                joints,
                scores,
                self.target_frames,
                self.training,
            )

            if self.training:
                if (
                    random.random()
                    < HORIZONTAL_FLIP_PROBABILITY
                ):
                    joints, scores = (
                        horizontal_flip(
                            joints,
                            scores,
                        )
                    )

                if JOINT_NOISE_SIGMA > 0:
                    joints = add_joint_noise(
                        joints,
                        scores,
                        JOINT_NOISE_SIGMA,
                    )

                if FRAME_DROP_PROBABILITY > 0:
                    joints, scores = (
                        random_drop_frames(
                            joints,
                            scores,
                            FRAME_DROP_PROBABILITY,
                        )
                    )

            motion = create_motion(
                joints,
                scores,
            )

            # x, y, confidence, dx, dy
            features = np.concatenate(
                [
                    joints,
                    scores[..., None],
                    motion,
                ],
                axis=-1,
            )

            # (T, M, V, C) -> (C, T, V, M)
            features = features.transpose(
                3,
                0,
                2,
                1,
            )

            label = self.class_to_id[
                row["label_name"]
            ]

            return (
                torch.from_numpy(
                    features.astype(np.float32)
                ),
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
            )

    # %% Cell 9
    train_dataset = BadmintonSTGCNDataset(
        train_df,
        class_to_id,
        target_frames=TARGET_FRAMES,
        training=True,
    )

    val_dataset = BadmintonSTGCNDataset(
        val_df,
        class_to_id,
        target_frames=TARGET_FRAMES,
        training=False,
    )

    test_dataset = BadmintonSTGCNDataset(
        test_df,
        class_to_id,
        target_frames=TARGET_FRAMES,
        training=False,
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": (
            NUM_WORKERS > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    features_batch, labels_batch = next(
        iter(train_loader)
    )

    print("Features:", features_batch.shape)
    print("Labels:", labels_batch.shape)
    print("Finite:", torch.isfinite(features_batch).all())

    assert features_batch.shape[1:] == (
        5,
        TARGET_FRAMES,
        17,
        2,
    )

    # %% Cell 10
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


    def show_skeleton(
        dataset,
        sample_index=0,
        frame_index=None,
    ):
        features, label = dataset[
            sample_index
        ]

        features = features.numpy()

        if frame_index is None:
            frame_index = (
                features.shape[1] // 2
            )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
        )

        for person_index, axis in enumerate(
            axes
        ):
            xy = features[
                :2,
                frame_index,
                :,
                person_index,
            ].T

            confidence = features[
                2,
                frame_index,
                :,
                person_index,
            ]

            for start, end in COCO_EDGES:
                if (
                    confidence[start] > 0
                    and confidence[end] > 0
                ):
                    axis.plot(
                        [
                            xy[start, 0],
                            xy[end, 0],
                        ],
                        [
                            xy[start, 1],
                            xy[end, 1],
                        ],
                        color="tab:blue",
                        linewidth=2,
                    )

            visible = confidence > 0

            axis.scatter(
                xy[visible, 0],
                xy[visible, 1],
                c=confidence[visible],
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=40,
            )

            axis.set_title(
                f"Person {person_index}"
            )

            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.grid(alpha=0.2)

        fig.suptitle(
            f"Label: {class_names[int(label)]}"
        )

        plt.tight_layout()
        plt.show()


    show_skeleton(
        val_dataset,
        sample_index=0,
    )

    # %% Cell 11
    from badminton_stroke.models.graphs import normalize_digraph, build_coco_adjacency
    ADJACENCY = build_coco_adjacency()
    print("Adjacency:", ADJACENCY.shape)

    # %% Cell 12
    from badminton_stroke.models.stgcn import SpatialGraphConv

    # %% Cell 13
    from badminton_stroke.models.stgcn import STGCNBlock

    # %% Cell 14
    from badminton_stroke.models.stgcn import STGCNV2

    # %% Cell 15
    model = STGCNV2(
        in_channels=5,
        num_classes=len(class_names),
        adjacency=ADJACENCY,
        dropout=0.4,
    ).to(DEVICE)

    with torch.no_grad():
        test_logits, test_attention = model(
            features_batch[:2].to(DEVICE),
            return_attention=True,
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Logits:", test_logits.shape)
    print("Attention:", test_attention)
    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )

    assert test_logits.shape == (
        2,
        len(class_names),
    )

    assert test_attention.shape == (
        2,
        2,
    )

    # %% Cell 16
    train_counts = (
        train_df["label_name"]
        .value_counts()
        .reindex(class_names)
        .to_numpy()
    )

    class_weights = (
        len(train_df)
        / (
            len(class_names)
            * train_counts
        )
    )

    # Không để lớp hiếm có trọng số quá lớn.
    class_weights = np.clip(
        class_weights,
        0.5,
        3.0,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Class weights:")

    display(
        pd.DataFrame(
            {
                "class": class_names,
                "train_count": train_counts,
                "weight": (
                    class_weights.cpu().numpy()
                ),
            }
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS,
        )
    )

    cosine_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=(
                NUM_EPOCHS
                - WARMUP_EPOCHS
            ),
            eta_min=1e-6,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[
                WARMUP_EPOCHS,
            ],
        )
    )

    # %% Cell 17
    def run_epoch(
        model,
        loader,
        criterion,
        optimizer=None,
    ):
        training = optimizer is not None

        model.train(training)

        total_loss = 0.0
        y_true = []
        y_pred = []

        progress = tqdm(
            loader,
            leave=False,
            desc=(
                "Train"
                if training
                else "Validation"
            ),
        )

        for features, labels in progress:
            features = features.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(
                training
            ):
                logits = model(features)

                loss = criterion(
                    logits,
                    labels,
                )

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    optimizer.step()

            total_loss += (
                loss.item()
                * len(labels)
            )

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                labels
                .detach()
                .cpu()
                .tolist()
            )

            y_pred.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

        return {
            "loss": (
                total_loss
                / len(loader.dataset)
            ),
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        }

    # %% Cell 18
    CHECKPOINT_PATH = (
        OUTPUT_ROOT
        / "stgcn_v2_best.pt"
    )

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        started = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        result = {
            "epoch": epoch,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "train_macro_f1": (
                train_metrics["macro_f1"]
            ),
            "val_loss": (
                val_metrics["loss"]
            ),
            "val_accuracy": (
                val_metrics["accuracy"]
            ),
            "val_macro_f1": (
                val_metrics["macro_f1"]
            ),
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "seconds": (
                time.time() - started
            ),
        }

        history.append(result)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={result['train_loss']:.4f}, "
            f"f1={result['train_macro_f1']:.4f}, "
            f"acc={result['train_accuracy']:.4f} | "
            f"val loss={result['val_loss']:.4f}, "
            f"f1={result['val_macro_f1']:.4f}, "
            f"acc={result['val_accuracy']:.4f} | "
            f"lr={result['learning_rate']:.2e} | "
            f"{result['seconds']:.1f}s"
        )

        if (
            result["val_macro_f1"]
            > best_val_f1
        ):
            best_val_f1 = (
                result["val_macro_f1"]
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "class_names": class_names,
                    "target_frames": (
                        TARGET_FRAMES
                    ),
                    "input_channels": 5,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": (
                        best_val_f1
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  Đã lưu checkpoint tốt nhất"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping tại epoch "
                f"{epoch}"
            )
            break


    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_ROOT
        / "training_history.csv",
        index=False,
    )

    print("Best epoch:", best_epoch)
    print(
        "Best validation Macro F1:",
        best_val_f1,
    )

    # %% Cell 19
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print(
        "Đã nạp best epoch:",
        checkpoint["best_epoch"],
    )

    print(
        "Best validation Macro F1:",
        checkpoint["best_val_macro_f1"],
    )

    # %% Cell 20
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4),
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train",
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation",
    )

    axes[0].set_title("Loss")

    axes[1].plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        label="Train",
    )

    axes[1].plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        label="Validation",
    )

    axes[1].set_title("Accuracy")

    axes[2].plot(
        history_df["epoch"],
        history_df["train_macro_f1"],
        label="Train",
    )

    axes[2].plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="Validation",
    )

    axes[2].set_title("Macro F1")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # %% Cell 21
    @torch.no_grad()
    def predict_dataset(
        model,
        loader,
    ):
        model.eval()

        y_true = []
        y_pred = []
        probabilities = []
        attention_values = []

        for features, labels in tqdm(
            loader,
            desc="Test",
        ):
            features = features.to(
                DEVICE,
                non_blocking=True,
            )

            logits, attention = model(
                features,
                return_attention=True,
            )

            probability = (
                logits
                .softmax(dim=1)
                .cpu()
            )

            y_true.extend(
                labels.tolist()
            )

            y_pred.extend(
                probability
                .argmax(dim=1)
                .tolist()
            )

            probabilities.append(
                probability
            )

            attention_values.append(
                attention.cpu()
            )

        return (
            np.asarray(y_true),
            np.asarray(y_pred),
            torch.cat(
                probabilities
            ).numpy(),
            torch.cat(
                attention_values
            ).numpy(),
        )


    (
        y_true,
        y_pred,
        y_probability,
        person_attention,
    ) = predict_dataset(
        model,
        test_loader,
    )

    test_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    test_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}"
    )

    # %% Cell 22
    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
        target_names=class_names,
        zero_division=0,
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    with open(
        OUTPUT_ROOT
        / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report_dict,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # %% Cell 23
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
        normalize="true",
    )

    plt.figure(
        figsize=(
            max(
                11,
                len(class_names) * 0.7,
            ),
            max(
                9,
                len(class_names) * 0.6,
            ),
        )
    )

    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title(
        "ST-GCN V2 normalized confusion matrix"
    )

    plt.xticks(
        rotation=70,
        ha="right",
    )

    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # %% Cell 24
    prediction_df = (
        test_df
        .reset_index(drop=True)
        .copy()
    )

    prediction_df["true_label"] = [
        class_names[index]
        for index in y_true
    ]

    prediction_df["predicted_label"] = [
        class_names[index]
        for index in y_pred
    ]

    prediction_df["confidence"] = (
        y_probability.max(axis=1)
    )

    prediction_df["correct"] = (
        y_true == y_pred
    )

    prediction_df["person_0_attention"] = (
        person_attention[:, 0]
    )

    prediction_df["person_1_attention"] = (
        person_attention[:, 1]
    )

    prediction_df.to_csv(
        OUTPUT_ROOT
        / "test_predictions.csv",
        index=False,
    )

    print("Dự đoán sai có confidence cao:")

    display(
        prediction_df[
            ~prediction_df["correct"]
        ]
        .sort_values(
            "confidence",
            ascending=False,
        )
        .head(20)
    )

    # %% Cell 25
    per_class_records = []

    for class_id, class_name in enumerate(
        class_names
    ):
        indices = y_true == class_id
        total = int(indices.sum())

        correct = int(
            (
                y_pred[indices]
                == class_id
            ).sum()
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        per_class_records.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )

    per_class_df = pd.DataFrame(
        per_class_records
    )

    display(
        per_class_df.sort_values(
            "accuracy"
        )
    )

    plt.figure(figsize=(13, 6))

    sns.barplot(
        data=per_class_df,
        x="class_name",
        y="accuracy",
    )

    plt.axhline(
        test_accuracy,
        color="red",
        linestyle="--",
        label=(
            f"Overall accuracy "
            f"{test_accuracy:.3f}"
        ),
    )

    plt.xticks(
        rotation=70,
        ha="right",
    )

    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("Recall/accuracy")
    plt.title("Accuracy theo từng lớp")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %% Cell 26
    confusion_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)
        ),
    )

    confusion_records = []

    for true_id in range(
        len(class_names)
    ):
        for predicted_id in range(
            len(class_names)
        ):
            if true_id == predicted_id:
                continue

            count = int(
                confusion_counts[
                    true_id,
                    predicted_id,
                ]
            )

            if count > 0:
                confusion_records.append(
                    {
                        "true_label": (
                            class_names[true_id]
                        ),
                        "predicted_label": (
                            class_names[predicted_id]
                        ),
                        "count": count,
                    }
                )

    confusion_pairs_df = pd.DataFrame(
        confusion_records
    ).sort_values(
        "count",
        ascending=False,
    )

    display(
        confusion_pairs_df.head(30)
    )

    # %% Cell 27
    metadata = {
        "model": "ST-GCN V2",
        "input_channels": [
            "normalized_x",
            "normalized_y",
            "confidence",
            "motion_x",
            "motion_y",
        ],
        "target_frames": TARGET_FRAMES,
        "num_joints": 17,
        "num_persons": 2,
        "num_classes": len(class_names),
        "class_names": class_names,
        "best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "best_val_macro_f1": float(
            checkpoint[
                "best_val_macro_f1"
            ]
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_macro_f1": float(
            test_macro_f1
        ),
        "seed": SEED,
    }

    with open(
        OUTPUT_ROOT
        / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Đã lưu kết quả tại:",
        OUTPUT_ROOT,
    )
