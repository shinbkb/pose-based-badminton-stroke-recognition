"""Exported from notebooks/stgcn_ablation.ipynb.

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

    SCORES_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "scores_npy_yolo26s_pose"
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

    assert SCORES_ROOT.exists(), (
        f"Không tìm thấy: {SCORES_ROOT}"
    )

    print("Joints root:", JOINTS_ROOT)
    print("Scores root:", SCORES_ROOT)

    # %% Cell 3
    EXPERIMENTS = {
        "joint_baseline": {
            "bone": False,
            "hitter_only": False,
            "position": False,
            "shuttle": False,
        },

        "joint_bone": {
            "bone": True,
            "hitter_only": False,
            "position": False,
            "shuttle": False,
        },

        "joint_hitter": {
            "bone": False,
            "hitter_only": True,
            "position": False,
            "shuttle": False,
        },

        "joint_position": {
            "bone": False,
            "hitter_only": False,
            "position": True,
            "shuttle": False,
        },

        "joint_shuttle": {
            "bone": False,
            "hitter_only": False,
            "position": False,
            "shuttle": True,
        },

        "joint_bone_position_shuttle": {
            "bone": True,
            "hitter_only": False,
            "position": True,
            "shuttle": True,
        },
    }


    # Đổi tên này để chạy từng thí nghiệm.
    EXPERIMENT_NAME = "joint_position"

    EXPERIMENT_CONFIG = EXPERIMENTS[
        EXPERIMENT_NAME
    ]

    ABLATION_ROOT = (
        EXPERIMENTS_DIR / "stgcn_ablation"
        / EXPERIMENT_NAME
    )

    ABLATION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SHUTTLE_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "shuttle_npy"
    )

    # Video của bạn hiện có kích thước này.
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    print("Experiment:", EXPERIMENT_NAME)
    print(EXPERIMENT_CONFIG)
    print("Output:", ABLATION_ROOT)

    # %% Cell 4
    MANIFEST_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "manifests"
    )


    def find_score_path(joint_path):
        relative_path = joint_path.relative_to(
            JOINTS_ROOT
        )

        score_filename = (
            relative_path.name.replace(
                "_joints.npy",
                "_scores.npy",
            )
        )

        score_path = (
            SCORES_ROOT
            / relative_path.parent
            / score_filename
        )

        if score_path.exists():
            return score_path

        return None


    def build_index(split):
        split_root = JOINTS_ROOT / split

        manifest_path = (
            MANIFEST_ROOT
            / f"{split}.csv"
        )

        assert split_root.exists(), split_root
        assert manifest_path.exists(), manifest_path

        manifest_df = pd.read_csv(
            manifest_path
        )

        manifest_columns = [
            "clip_id",
            "player",
            "video_path",
            "class_name",
        ]

        manifest_df = manifest_df[
            manifest_columns
        ].copy()

        records = []

        joint_files = sorted(
            split_root.rglob("*_joints.npy")
        )

        for joint_path in joint_files:
            relative_path = (
                joint_path.relative_to(
                    split_root
                )
            )

            if len(relative_path.parts) < 2:
                raise RuntimeError(
                    f"{joint_path} không nằm trong "
                    "thư mục class."
                )

            label_name = relative_path.parts[0]

            clip_id = joint_path.name.removesuffix(
                "_joints.npy"
            )

            score_path = find_score_path(
                joint_path
            )

            shuttle_path = (
                SHUTTLE_ROOT
                / split
                / label_name
                / f"{clip_id}_shuttle.npy"
            )

            records.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "label_name": label_name,
                    "joint_path": str(joint_path),
                    "score_path": (
                        str(score_path)
                        if score_path is not None
                        else ""
                    ),
                    "shuttle_path": str(
                        shuttle_path
                    ),
                }
            )

        index_df = pd.DataFrame(records)

        assert len(index_df) > 0, (
            f"Không tìm thấy joints trong {split_root}"
        )

        index_df = index_df.merge(
            manifest_df,
            on="clip_id",
            how="left",
            suffixes=("", "_manifest"),
            validate="one_to_one",
        )

        assert index_df["player"].notna().all(), (
            f"Có clip trong {split} không ghép "
            "được với manifest"
        )

        missing_scores = index_df[
            "score_path"
        ].eq("")

        if missing_scores.any():
            examples = index_df.loc[
                missing_scores,
                "joint_path",
            ].head(5).tolist()

            raise FileNotFoundError(
                f"Thiếu {missing_scores.sum()} score file "
                f"trong split {split}. Ví dụ: {examples}"
            )

        return index_df


    train_df = build_index("train")
    val_df = build_index("val")
    test_df = build_index("test")

    class_names = sorted(
        train_df["label_name"]
        .unique()
        .tolist()
    )

    class_to_id = {
        name: index
        for index, name
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

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print(
        "Score files:",
        train_df["score_path"].ne("").sum(),
        "/",
        len(train_df),
    )

    display(
        train_df[
            [
                "clip_id",
                "label_name",
                "player",
                "joint_path",
                "score_path",
            ]
        ].head()
    )

    # %% Cell 5
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

    # %% Cell 6
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

    # %% Cell 7
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

    # %% Cell 8
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

    # %% Cell 9
    COCO_BONES = [
        # Torso
        (11, 12),
        (11, 5),
        (5, 6),

        # Left arm
        (5, 7),
        (7, 9),

        # Right arm
        (6, 8),
        (8, 10),

        # Left leg
        (11, 13),
        (13, 15),

        # Right leg
        (12, 14),
        (14, 16),

        # Head
        (5, 0),
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 4),
    ]

    ARM_JOINTS = [
        5, 6,       # Shoulder
        7, 8,       # Elbow
        9, 10,      # Wrist
    ]

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


    def create_bone_features(
        joints,
        scores,
    ):
        """
        joints: (T, M, 17, 2)
        scores: (T, M, 17)

        Bone được đặt tại joint con để vẫn giữ
        graph COCO gồm 17 node.
        """

        bones = np.zeros_like(
            joints,
            dtype=np.float32,
        )

        bone_scores = np.zeros_like(
            scores,
            dtype=np.float32,
        )

        for parent, child in COCO_BONES:
            valid = (
                (scores[:, :, parent] > 0)
                & (scores[:, :, child] > 0)
            )

            vector = (
                joints[:, :, child]
                - joints[:, :, parent]
            )

            bones[:, :, child] = np.where(
                valid[..., None],
                vector,
                0.0,
            )

            bone_scores[:, :, child] = np.where(
                valid,
                np.minimum(
                    scores[:, :, parent],
                    scores[:, :, child],
                ),
                0.0,
            )

        bone_motion = create_motion(
            bones,
            bone_scores,
        )

        # bx, by, dbx, dby
        return np.concatenate(
            [
                bones,
                bone_motion,
            ],
            axis=-1,
        )


    def extract_player_positions(
        raw_joints,
        raw_scores,
        frame_width,
        frame_height,
        confidence_threshold,
    ):
        """
        Kết quả: (T, M, 3)
        [position_x, position_y, valid]

        Tọa độ được đưa về gần [-0.5, 0.5].
        Đây mới là tọa độ ảnh, chưa phải court
        homography như bài BST.
        """

        num_frames, num_people = (
            raw_joints.shape[:2]
        )

        positions = np.zeros(
            (num_frames, num_people, 3),
            dtype=np.float32,
        )

        for frame_index in range(num_frames):
            for person_index in range(
                num_people
            ):
                ankle_valid = (
                    raw_scores[
                        frame_index,
                        person_index,
                        [LEFT_ANKLE, RIGHT_ANKLE],
                    ]
                    >= confidence_threshold
                )

                if ankle_valid.all():
                    point = raw_joints[
                        frame_index,
                        person_index,
                        [LEFT_ANKLE, RIGHT_ANKLE],
                    ].mean(axis=0)

                else:
                    # Fallback bằng tâm hai hông.
                    hip_valid = (
                        raw_scores[
                            frame_index,
                            person_index,
                            [LEFT_HIP, RIGHT_HIP],
                        ]
                        >= confidence_threshold
                    )

                    if not hip_valid.all():
                        continue

                    point = raw_joints[
                        frame_index,
                        person_index,
                        [LEFT_HIP, RIGHT_HIP],
                    ].mean(axis=0)

                positions[
                    frame_index,
                    person_index,
                    0,
                ] = (
                    point[0] / frame_width
                    - 0.5
                )

                positions[
                    frame_index,
                    person_index,
                    1,
                ] = (
                    point[1] / frame_height
                    - 0.5
                )

                positions[
                    frame_index,
                    person_index,
                    2,
                ] = 1.0

        return positions


    def select_hitter_by_motion(
        joints,
        scores,
    ):
        """
        Hitter proxy:
        chọn người có chuyển động tay lớn nhất.

        Đây là heuristic để thử nghiệm, chưa phải
        nhãn hitter chính xác.
        """

        motion = create_motion(
            joints,
            scores,
        )

        valid = (
            scores[:, :, ARM_JOINTS] > 0
        )

        arm_motion = motion[
            :, :, ARM_JOINTS
        ]

        energy = np.linalg.norm(
            arm_motion,
            axis=-1,
        )

        energy = (
            energy * valid
        ).sum(axis=(0, 2))

        if np.allclose(energy, 0):
            visibility = (
                scores > 0
            ).sum(axis=(0, 2))

            return int(
                np.argmax(visibility)
            )

        return int(np.argmax(energy))

    # %% Cell 10
    def crop_or_pad_aligned(
        arrays,
        target_frames,
        training,
    ):
        """
        Tất cả array phải có chiều đầu tiên là T.
        """

        num_frames = len(arrays[0])

        assert all(
            len(array) == num_frames
            for array in arrays
        )

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

            return tuple(
                array[start:end]
                for array in arrays
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

        outputs = []

        for array in arrays:
            output = np.zeros(
                (
                    target_frames,
                    *array.shape[1:],
                ),
                dtype=array.dtype,
            )

            output[start:end] = array
            outputs.append(output)

        return tuple(outputs)


    def random_drop_aligned(
        arrays,
        scores,
        probability,
    ):
        arrays = [
            array.copy()
            for array in arrays
        ]

        scores = scores.copy()

        for frame_index in range(
            1,
            len(scores),
        ):
            has_pose = np.any(
                scores[frame_index] > 0
            )

            if (
                has_pose
                and random.random()
                < probability
            ):
                for array in arrays:
                    array[frame_index] = (
                        array[frame_index - 1]
                    )

                scores[frame_index] = (
                    scores[frame_index - 1]
                )

        return arrays, scores

    # %% Cell 11
    def tracknet_csv_to_npy(
        csv_path,
        output_path,
        frame_count,
        frame_width=1280,
        frame_height=720,
    ):
        frame = pd.read_csv(csv_path)

        column_map = {
            column.lower(): column
            for column in frame.columns
        }

        frame_column = column_map.get(
            "frame",
            column_map.get("frame_id"),
        )

        x_column = column_map.get(
            "x",
            column_map.get("x-coordinate"),
        )

        y_column = column_map.get(
            "y",
            column_map.get("y-coordinate"),
        )

        visibility_column = column_map.get(
            "visibility",
            column_map.get("confidence"),
        )

        if x_column is None or y_column is None:
            raise ValueError(
                f"Không tìm thấy cột X/Y: "
                f"{frame.columns.tolist()}"
            )

        shuttle = np.zeros(
            (frame_count, 3),
            dtype=np.float32,
        )

        for row_index, row in (
            frame.iterrows()
        ):
            frame_index = (
                int(row[frame_column])
                if frame_column is not None
                else row_index
            )

            if not (
                0 <= frame_index < frame_count
            ):
                continue

            x = float(row[x_column])
            y = float(row[y_column])

            confidence = (
                float(row[visibility_column])
                if visibility_column is not None
                else 1.0
            )

            if (
                confidence <= 0
                or x <= 0
                or y <= 0
            ):
                continue

            shuttle[frame_index, 0] = (
                x / frame_width - 0.5
            )

            shuttle[frame_index, 1] = (
                y / frame_height - 0.5
            )

            shuttle[frame_index, 2] = (
                confidence
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            shuttle,
        )

    # %% Cell 12
    class BadmintonAblationDataset(Dataset):
        def __init__(
            self,
            index_df,
            class_to_id,
            experiment_config,
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
            self.config = experiment_config

            self.target_frames = (
                target_frames
            )

            self.confidence_threshold = (
                confidence_threshold
            )

            self.training = training

        def __len__(self):
            return len(self.index_df)

        def _load_shuttle(
            self,
            row,
            num_frames,
        ):
            if not self.config["shuttle"]:
                return np.zeros(
                    (num_frames, 3),
                    dtype=np.float32,
                )

            shuttle_path = Path(
                row["shuttle_path"]
            )

            if not shuttle_path.exists():
                raise FileNotFoundError(
                    f"Thiếu shuttle file: "
                    f"{shuttle_path}"
                )

            shuttle = np.load(
                shuttle_path
            ).astype(np.float32)

            if shuttle.shape != (
                num_frames,
                3,
            ):
                raise ValueError(
                    f"Shuttle shape sai: "
                    f"{shuttle.shape}, "
                    f"mong đợi {(num_frames, 3)}"
                )

            return shuttle

        def __getitem__(self, index):
            row = self.index_df.iloc[index]

            raw_joints = np.load(
                row["joint_path"]
            ).astype(np.float32)

            if (
                raw_joints.ndim != 4
                or raw_joints.shape[1:]
                != (2, 17, 2)
            ):
                raise ValueError(
                    f"Joint shape sai: "
                    f"{raw_joints.shape}"
                )

            raw_scores = load_scores(
                row["score_path"],
                raw_joints,
            )

            positions = (
                extract_player_positions(
                    raw_joints,
                    raw_scores,
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                    self.confidence_threshold,
                )
            )

            shuttle = self._load_shuttle(
                row,
                len(raw_joints),
            )

            joints, scores = normalize_skeleton(
                raw_joints,
                raw_scores,
                self.confidence_threshold,
            )

            hitter_index = None

            if self.config["hitter_only"]:
                # Chọn trên toàn bộ chuỗi trước random crop
                # và augmentation để hitter ổn định giữa epoch.
                hitter_index = select_hitter_by_motion(
                    joints,
                    scores,
                )

            (
                joints,
                scores,
                positions,
                shuttle,
            ) = crop_or_pad_aligned(
                [
                    joints,
                    scores,
                    positions,
                    shuttle,
                ],
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

                    # Position đã centered.
                    positions[..., 0] *= -1

                    if self.config["shuttle"]:
                        shuttle[..., 0] *= -1

                if JOINT_NOISE_SIGMA > 0:
                    joints = add_joint_noise(
                        joints,
                        scores,
                        JOINT_NOISE_SIGMA,
                    )

                if FRAME_DROP_PROBABILITY > 0:
                    aligned, scores = (
                        random_drop_aligned(
                            [
                                joints,
                                positions,
                                shuttle,
                            ],
                            scores,
                            FRAME_DROP_PROBABILITY,
                        )
                    )

                    (
                        joints,
                        positions,
                        shuttle,
                    ) = aligned

            joint_motion = create_motion(
                joints,
                scores,
            )

            feature_parts = [
                # x, y
                joints,

                # confidence
                scores[..., None],

                # dx, dy
                joint_motion,
            ]

            if self.config["bone"]:
                bone_features = (
                    create_bone_features(
                        joints,
                        scores,
                    )
                )

                feature_parts.append(
                    bone_features
                )

            if self.config["position"]:
                # (T, M, 3)
                # -> (T, M, 17, 3)
                position_features = np.repeat(
                    positions[:, :, None, :],
                    repeats=17,
                    axis=2,
                )

                feature_parts.append(
                    position_features
                )

            if self.config["shuttle"]:
                shuttle_xy = shuttle[:, :2]
                shuttle_confidence = shuttle[:, 2]

                shuttle_motion = np.zeros_like(
                    shuttle_xy,
                    dtype=np.float32,
                )

                valid_pair = (
                    (shuttle_confidence[1:] > 0)
                    & (shuttle_confidence[:-1] > 0)
                )

                shuttle_motion[1:] = np.where(
                    valid_pair[:, None],
                    (
                        shuttle_xy[1:]
                        - shuttle_xy[:-1]
                    ),
                    0.0,
                )

                shuttle_features = np.concatenate(
                    [
                        shuttle_xy,
                        shuttle_confidence[
                            :, None
                        ],
                        shuttle_motion,
                    ],
                    axis=-1,
                )
                # (T, 5)

                shuttle_features = (
                    shuttle_features[
                        :, None, None, :
                    ]
                )

                shuttle_features = np.repeat(
                    shuttle_features,
                    repeats=joints.shape[1],
                    axis=1,
                )

                shuttle_features = np.repeat(
                    shuttle_features,
                    repeats=17,
                    axis=2,
                )

                feature_parts.append(
                    shuttle_features
                )

            features = np.concatenate(
                feature_parts,
                axis=-1,
            )
            # (T, M, V, C)

            if self.config["hitter_only"]:
                # Chỉ còn một person.
                features = features[
                    :,
                    hitter_index:
                    hitter_index + 1,
                ]

            # (T, M, V, C)
            # -> (C, T, V, M)
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

    # %% Cell 13
    train_dataset = BadmintonAblationDataset(
        train_df,
        class_to_id,
        EXPERIMENT_CONFIG,
        target_frames=TARGET_FRAMES,
        training=True,
    )

    val_dataset = BadmintonAblationDataset(
        val_df,
        class_to_id,
        EXPERIMENT_CONFIG,
        target_frames=TARGET_FRAMES,
        training=False,
    )

    test_dataset = BadmintonAblationDataset(
        test_df,
        class_to_id,
        EXPERIMENT_CONFIG,
        target_frames=TARGET_FRAMES,
        training=False,
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (
            DEVICE.type == "cuda"
        ),
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

    INPUT_CHANNELS = (
        features_batch.shape[1]
    )

    NUM_PERSONS = (
        features_batch.shape[-1]
    )

    print("Experiment:", EXPERIMENT_NAME)
    print("Features:", features_batch.shape)
    print("Input channels:", INPUT_CHANNELS)
    print("Persons:", NUM_PERSONS)
    print(
        "Finite:",
        torch.isfinite(
            features_batch
        ).all().item(),
    )

    # %% Cell 14
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

        num_people = features.shape[-1]

        fig, axes = plt.subplots(
            1,
            num_people,
            figsize=(5 * num_people, 5),
            squeeze=False,
        )

        axes = axes[0]

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

    # %% Cell 15
    from badminton_stroke.models.graphs import normalize_digraph, build_coco_adjacency
    ADJACENCY = build_coco_adjacency()
    print("Adjacency:", ADJACENCY.shape)

    # %% Cell 16
    from badminton_stroke.models.stgcn import SpatialGraphConv

    # %% Cell 17
    from badminton_stroke.models.stgcn import STGCNBlock

    # %% Cell 18
    from badminton_stroke.models.stgcn import STGCNV2

    # %% Cell 19
    model = STGCNV2(
        in_channels=INPUT_CHANNELS,
        num_classes=len(class_names),
        adjacency=ADJACENCY,
        dropout=0.4,
    ).to(DEVICE)

    with torch.no_grad():
        logits, attention = model(
            features_batch[:2].to(DEVICE),
            return_attention=True,
        )

    print("Logits:", logits.shape)
    print("Attention:", attention)
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    # %% Cell 20
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

    # %% Cell 21
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
                labels=np.arange(
                    len(class_names)
                ),
                average="macro",
                zero_division=0,
            ),
        }

    # %% Cell 22
    CHECKPOINT_PATH = (
        ABLATION_ROOT
        / "best.pt"
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
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics[
                "accuracy"
            ],
            "train_macro_f1": train_metrics[
                "macro_f1"
            ],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_macro_f1": val_metrics[
                "macro_f1"
            ],
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "seconds": time.time() - started,
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

        if result["val_macro_f1"] > best_val_f1:
            best_val_f1 = result[
                "val_macro_f1"
            ]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "experiment_name": (
                        EXPERIMENT_NAME
                    ),
                    "experiment_config": (
                        EXPERIMENT_CONFIG
                    ),
                    "class_names": class_names,
                    "target_frames": (
                        TARGET_FRAMES
                    ),
                    "input_channels": (
                        INPUT_CHANNELS
                    ),
                    "num_persons": (
                        NUM_PERSONS
                    ),
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
        ABLATION_ROOT
        / "training_history.csv",
        index=False,
    )

    print("Best epoch:", best_epoch)
    print(
        "Best validation Macro F1:",
        best_val_f1,
    )

    # %% Cell 23
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

    # %% Cell 24
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
        ABLATION_ROOT
        / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # %% Cell 25
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
        labels=np.arange(
            len(class_names)
        ),
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

    # %% Cell 26
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
        ABLATION_ROOT
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

    # %% Cell 27
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
        ABLATION_ROOT
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    # %% Cell 28
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

    for person_index in range(
        person_attention.shape[1]
    ):
        prediction_df[
            f"person_{person_index}_attention"
        ] = person_attention[
            :, person_index
        ]

    prediction_df.to_csv(
        ABLATION_ROOT
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

    # %% Cell 29
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

    # %% Cell 30
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

    # %% Cell 31
    input_feature_names = [
        "normalized_x",
        "normalized_y",
        "confidence",
        "motion_x",
        "motion_y",
    ]

    if EXPERIMENT_CONFIG["bone"]:
        input_feature_names += [
            "bone_x",
            "bone_y",
            "bone_motion_x",
            "bone_motion_y",
        ]

    if EXPERIMENT_CONFIG["position"]:
        input_feature_names += [
            "position_x",
            "position_y",
            "position_valid",
        ]

    if EXPERIMENT_CONFIG["shuttle"]:
        input_feature_names += [
            "shuttle_x",
            "shuttle_y",
            "shuttle_confidence",
            "shuttle_motion_x",
            "shuttle_motion_y",
        ]

    assert len(input_feature_names) == (
        INPUT_CHANNELS
    )

    metadata = {
        "model": "ST-GCN V2",
        "experiment": EXPERIMENT_NAME,
        "experiment_config": (
            EXPERIMENT_CONFIG
        ),
        "input_features": (
            input_feature_names
        ),
        "input_channels": INPUT_CHANNELS,
        "target_frames": TARGET_FRAMES,
        "num_joints": 17,
        "num_persons": NUM_PERSONS,
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
        ABLATION_ROOT
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
        ABLATION_ROOT,
    )

    # %% Cell 32
    summary = {
        "experiment": EXPERIMENT_NAME,
        "input_channels": INPUT_CHANNELS,
        "num_persons": NUM_PERSONS,
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
        "test_weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    with open(
        ABLATION_ROOT / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    display(
        pd.DataFrame([summary])
    )

    # %% Cell 33
    ABLATION_BASE = (
        EXPERIMENTS_DIR / "stgcn_ablation"
    )

    records = []

    for summary_path in sorted(
        ABLATION_BASE.glob(
            "*/summary.json"
        )
    ):
        with open(
            summary_path,
            encoding="utf-8",
        ) as file:
            records.append(
                json.load(file)
            )

    if records:
        comparison_df = pd.DataFrame(
            records
        ).sort_values(
            "test_macro_f1",
            ascending=False,
        )

        display(comparison_df)

        comparison_df.to_csv(
            ABLATION_BASE
            / "ablation_comparison.csv",
            index=False,
        )
    else:
        comparison_df = pd.DataFrame()
        print(
            "Chưa có summary.json. "
            "Hãy chạy ít nhất một thí nghiệm."
        )
