"""Exported from notebooks/yolov26pose.ipynb.

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
    import torch
    import ultralytics

    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Compute capability:",
            torch.cuda.get_device_capability(0),
        )

    # %% Cell 2
    import torch

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU capability:", torch.cuda.get_device_capability(0))
    print("Architectures:", torch.cuda.get_arch_list())

    # %% [markdown]
    # ## 3. Khởi tạo YOLO26 Pose

    # %% Cell 4
    from ultralytics import YOLO

    pose_model = YOLO(str(MODELS_DIR / "yolo26s-pose.pt"))

    # %% Cell 5
    from pathlib import Path
    import pandas as pd

    from badminton_stroke.paths import PROJECT_ROOT

    MANIFEST_DIR = (
        PROJECT_ROOT
        / "processed"
        / "manifests"
    )

    manifest_parts = []

    for split in ["train", "val", "test"]:
        manifest_path = (
            MANIFEST_DIR / f"{split}.csv"
        )

        assert manifest_path.exists(), (
            f"Không tìm thấy: {manifest_path}"
        )

        split_df = pd.read_csv(manifest_path)

        # Bảo đảm mỗi dòng có thông tin split.
        split_df["split"] = split

        manifest_parts.append(split_df)

    manifest_df = pd.concat(
        manifest_parts,
        ignore_index=True,
    )

    print("Số video:", len(manifest_df))
    print("\nSố video theo split:")
    print(manifest_df["split"].value_counts())

    print("\nCác cột:")
    print(manifest_df.columns.tolist())

    display(manifest_df.head())

    # %% Cell 6
    OLD_PROJECT_ROOT = (
        "/home/dutai/khacbinh/pbl5_doantotnghiep"
    )

    manifest_df["video_path"] = (
        manifest_df["video_path"]
        .astype(str)
        .str.replace(
            OLD_PROJECT_ROOT,
            str(PROJECT_ROOT),
            regex=False,
        )
    )

    manifest_df["video_exists"] = (
        manifest_df["video_path"]
        .map(lambda path: Path(path).is_file())
    )

    print(
        manifest_df["video_exists"]
        .value_counts(dropna=False)
    )

    # %% Cell 7
    missing_videos = manifest_df.loc[
        ~manifest_df["video_exists"],
        ["video_path", "split"],
    ]

    print("Video không tồn tại:", len(missing_videos))
    display(missing_videos)

    manifest_df = (
        manifest_df[
            manifest_df["video_exists"]
        ]
        .drop(columns="video_exists")
        .reset_index(drop=True)
    )

    print("Số video sẽ xử lý:", len(manifest_df))

    # %% [markdown]
    # ## 4. Thử trên một frame

    # %% Cell 9
    import cv2

    sample_path = manifest_df.iloc[0]["video_path"]

    capture = cv2.VideoCapture(str(sample_path))
    opened, frame = capture.read()
    capture.release()

    assert opened and frame is not None, (
        f"Không đọc được video: {sample_path}"
    )

    results = pose_model.predict(
        source=frame,
        device=0,
        imgsz=640,
        conf=0.25,
        quantize=16,
        verbose=False,
    )

    result = results[0]

    print("Frame:", frame.shape)
    print("Số người:", len(result.boxes))

    if result.keypoints is not None:
        print(
            "Keypoints:",
            result.keypoints.xy.shape,
        )
        print(
            "Keypoint data:",
            result.keypoints.data.shape,
        )

    # %% [markdown]
    # ## 5. Hiển thị kết quả

    # %% Cell 11
    import matplotlib.pyplot as plt
    import cv2

    annotated_frame = result.plot()

    plt.figure(figsize=(14, 8))
    plt.imshow(
        cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB,
        )
    )
    plt.axis("off")
    plt.show()

    # %% [markdown]
    # ## 6. Chuyển kết quả về hai người Top và Bottom

    # %% Cell 13
    import numpy as np


    def yolo_result_to_two_players(
        result,
        frame_height,
        keypoint_threshold=0.25,
    ):
        joints = np.zeros(
            (2, 17, 2),
            dtype=np.float32,
        )

        scores = np.zeros(
            (2, 17),
            dtype=np.float32,
        )

        if (
            result.boxes is None
            or result.keypoints is None
            or len(result.boxes) == 0
        ):
            return joints, scores

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        keypoint_data = (
            result.keypoints.data
            .detach()
            .cpu()
            .numpy()
        )

        xy = keypoint_data[:, :, :2]

        if keypoint_data.shape[2] >= 3:
            confidence = keypoint_data[:, :, 2]
        else:
            confidence = np.ones(
                xy.shape[:2],
                dtype=np.float32,
            )

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Giữ tối đa hai người có bounding box lớn nhất.
        selected = np.argsort(areas)[::-1][:2]

        boxes = boxes[selected]
        xy = xy[selected]
        confidence = confidence[selected]

        center_y = (
            boxes[:, 1] + boxes[:, 3]
        ) / 2

        # Sắp xếp Top trước, Bottom sau.
        order = np.argsort(center_y)

        boxes = boxes[order]
        xy = xy[order]
        confidence = confidence[order]
        center_y = center_y[order]

        if len(xy) == 2:
            joints[0] = xy[0]
            joints[1] = xy[1]

            scores[0] = confidence[0]
            scores[1] = confidence[1]

        elif len(xy) == 1:
            # Chỉ phát hiện một người:
            # đưa vào Top hoặc Bottom theo vị trí dọc.
            player_slot = (
                0
                if center_y[0] < frame_height / 2
                else 1
            )

            joints[player_slot] = xy[0]
            scores[player_slot] = confidence[0]

        invalid = scores < keypoint_threshold
        joints[invalid] = 0.0

        return joints, scores

    # %% Cell 14
    joints, scores = yolo_result_to_two_players(
        result,
        frame_height=frame.shape[0],
    )

    print("Joints:", joints.shape)
    print("Scores:", scores.shape)
    print("Top valid joints:", np.count_nonzero(scores[0]))
    print("Bottom valid joints:", np.count_nonzero(scores[1]))

    # %% [markdown]
    # ## 7. Hàm xử lý một video

    # %% Cell 16
    def extract_video_joints_yolo26(
        video_path,
        model,
        image_size=640,
        person_confidence=0.25,
        keypoint_threshold=0.25,
    ):
        video_joints = []
        video_scores = []

        predictions = model.predict(
            source=str(video_path),
            stream=True,
            device=0,
            imgsz=image_size,
            conf=person_confidence,
            quantize=16,
            verbose=False,
        )

        for result in predictions:
            frame_height = result.orig_shape[0]

            frame_joints, frame_scores = (
                yolo_result_to_two_players(
                    result=result,
                    frame_height=frame_height,
                    keypoint_threshold=(
                        keypoint_threshold
                    ),
                )
            )

            video_joints.append(frame_joints)
            video_scores.append(frame_scores)

        if not video_joints:
            return (
                np.empty(
                    (0, 2, 17, 2),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 2, 17),
                    dtype=np.float32,
                ),
            )

        return (
            np.stack(video_joints).astype(
                np.float32
            ),
            np.stack(video_scores).astype(
                np.float32
            ),
        )

    # %% [markdown]
    # ## 8. Thử trên một video

    # %% Cell 18
    sample_row = manifest_df.iloc[0]

    sample_joints, sample_scores = (
        extract_video_joints_yolo26(
            video_path=sample_row["video_path"],
            model=pose_model,
        )
    )

    print("Video:", sample_row["video_path"])
    print("Joints:", sample_joints.shape)
    print("Scores:", sample_scores.shape)

    valid_frames = np.any(
        sample_joints != 0,
        axis=(1, 2, 3),
    )

    print(
        "Tỉ lệ frame hợp lệ:",
        valid_frames.mean(),
    )

    # %% [markdown]
    # ## 9. Benchmark trước khi chạy toàn bộ

    # %% Cell 20
    from time import perf_counter

    sample_manifest = manifest_df.head(10)

    start_time = perf_counter()
    total_frames = 0

    for row in sample_manifest.itertuples(
        index=False
    ):
        joints, scores = (
            extract_video_joints_yolo26(
                row.video_path,
                pose_model,
            )
        )

        total_frames += len(joints)

    elapsed = perf_counter() - start_time

    print("Số video:", len(sample_manifest))
    print("Số frame:", total_frames)
    print("Thời gian:", round(elapsed, 2), "giây")
    print(
        "Tốc độ:",
        round(total_frames / elapsed, 2),
        "FPS",
    )

    # %% Cell 21
    from pathlib import Path
    import pandas as pd
    import numpy as np

    from badminton_stroke.paths import PROJECT_ROOT

    YOLO_JOINTS_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "joints_npy_yolo26s_pose"
    )

    YOLO_SCORES_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "scores_npy_yolo26s_pose"
    )

    STATUS_PATH = (
        PROJECT_ROOT
        / "processed"
        / "yolo26s_pose_extraction_status.csv"
    )

    YOLO_JOINTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    YOLO_SCORES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Số video cần xử lý:", len(manifest_df))
    print("Joint output:", YOLO_JOINTS_ROOT)
    print("Score output:", YOLO_SCORES_ROOT)
    print("Status:", STATUS_PATH)

    # %% Cell 22
    import re


    def safe_directory_name(value):
        value = str(value).strip()

        return re.sub(
            r'[<>:"/\\|?*]+',
            "_",
            value,
        )


    def get_yolo_output_paths(row):
        split = str(row["split"])

        class_directory = (
            f"{int(row['class_id']):02d}_"
            f"{safe_directory_name(row['class_name'])}"
        )

        joint_directory = (
            YOLO_JOINTS_ROOT
            / split
            / class_directory
        )

        score_directory = (
            YOLO_SCORES_ROOT
            / split
            / class_directory
        )

        joint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        score_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        clip_id = str(row["clip_id"])

        joint_path = (
            joint_directory
            / f"{clip_id}_joints.npy"
        )

        score_path = (
            score_directory
            / f"{clip_id}_scores.npy"
        )

        return joint_path, score_path

    # %% Cell 23
    def atomic_save_npy(path, array):
        temporary_path = path.with_name(
            path.stem + ".part.npy"
        )

        np.save(
            temporary_path,
            array,
        )

        temporary_path.replace(path)

    # %% Cell 24
    if STATUS_PATH.exists():
        old_status_df = pd.read_csv(
            STATUS_PATH
        )

        status_records = {
            str(record["clip_id"]): record
            for record in old_status_df.to_dict(
                orient="records"
            )
        }

        print(
            "Đã đọc status cũ:",
            len(status_records),
        )
    else:
        status_records = {}
        print("Bắt đầu status mới.")


    def save_yolo_status():
        if not status_records:
            return

        status_df = pd.DataFrame(
            status_records.values()
        )

        status_df.to_csv(
            STATUS_PATH,
            index=False,
        )

    # %% Cell 25
    def extract_video_joints_yolo26(
        video_path,
        model,
        image_size=640,
        person_confidence=0.25,
        keypoint_threshold=0.25,
    ):
        video_joints = []
        video_scores = []

        predictions = model.predict(
            source=str(video_path),
            stream=True,
            device=0,
            imgsz=image_size,
            conf=person_confidence,
            quantize=16,
            verbose=False,
        )

        for result in predictions:
            frame_height = result.orig_shape[0]

            frame_joints, frame_scores = (
                yolo_result_to_two_players(
                    result=result,
                    frame_height=frame_height,
                    keypoint_threshold=(
                        keypoint_threshold
                    ),
                )
            )

            video_joints.append(frame_joints)
            video_scores.append(frame_scores)

        if not video_joints:
            return (
                np.empty(
                    (0, 2, 17, 2),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 2, 17),
                    dtype=np.float32,
                ),
            )

        return (
            np.stack(video_joints).astype(
                np.float32
            ),
            np.stack(video_scores).astype(
                np.float32
            ),
        )

    # %% Cell 26
    from time import perf_counter
    from tqdm import tqdm

    start_time = perf_counter()
    consecutive_errors = 0

    for index, (_, row) in enumerate(
        tqdm(
            manifest_df.iterrows(),
            total=len(manifest_df),
            desc="YOLO26 Pose extraction",
        )
    ):
        clip_id = str(row["clip_id"])

        joint_path, score_path = (
            get_yolo_output_paths(row)
        )

        # Resume: bỏ qua nếu cả hai file đã tồn tại
        # và có cấu trúc hợp lệ.
        if joint_path.exists() and score_path.exists():
            try:
                old_joints = np.load(
                    joint_path,
                    mmap_mode="r",
                )

                old_scores = np.load(
                    score_path,
                    mmap_mode="r",
                )

                valid_existing = (
                    old_joints.ndim == 4
                    and old_joints.shape[1:]
                    == (2, 17, 2)
                    and old_scores.ndim == 3
                    and old_scores.shape[1:]
                    == (2, 17)
                    and len(old_joints)
                    == len(old_scores)
                    and len(old_joints) > 0
                )

                if valid_existing:
                    if clip_id not in status_records:
                        valid_frames = np.any(
                            old_joints != 0,
                            axis=(1, 2, 3),
                        )

                        status_records[clip_id] = {
                            "clip_id": clip_id,
                            "split": row["split"],
                            "class_id": int(
                                row["class_id"]
                            ),
                            "class_name": row[
                                "class_name"
                            ],
                            "status": "success",
                            "num_frames": len(
                                old_joints
                            ),
                            "valid_ratio": float(
                                valid_frames.mean()
                            ),
                            "joint_path": str(
                                joint_path
                            ),
                            "score_path": str(
                                score_path
                            ),
                            "error": "",
                        }

                    consecutive_errors = 0
                    continue

            except Exception:
                # File cũ không hợp lệ sẽ được tạo lại.
                pass

        try:
            joints, scores = (
                extract_video_joints_yolo26(
                    video_path=row["video_path"],
                    model=pose_model,
                )
            )

            if len(joints) == 0:
                raise RuntimeError(
                    "Video không có frame."
                )

            if joints.shape[1:] != (2, 17, 2):
                raise RuntimeError(
                    "Sai shape joints: "
                    f"{joints.shape}"
                )

            if scores.shape[1:] != (2, 17):
                raise RuntimeError(
                    "Sai shape scores: "
                    f"{scores.shape}"
                )

            atomic_save_npy(
                joint_path,
                joints,
            )

            atomic_save_npy(
                score_path,
                scores,
            )

            valid_frames = np.any(
                joints != 0,
                axis=(1, 2, 3),
            )

            top_valid = np.any(
                joints[:, 0] != 0,
                axis=(1, 2),
            )

            bottom_valid = np.any(
                joints[:, 1] != 0,
                axis=(1, 2),
            )

            status_records[clip_id] = {
                "clip_id": clip_id,
                "split": row["split"],
                "class_id": int(
                    row["class_id"]
                ),
                "class_name": row["class_name"],
                "status": "success",
                "num_frames": len(joints),
                "valid_ratio": float(
                    valid_frames.mean()
                ),
                "top_valid_ratio": float(
                    top_valid.mean()
                ),
                "bottom_valid_ratio": float(
                    bottom_valid.mean()
                ),
                "joint_path": str(joint_path),
                "score_path": str(score_path),
                "error": "",
            }

            consecutive_errors = 0

        except Exception as error:
            consecutive_errors += 1

            status_records[clip_id] = {
                "clip_id": clip_id,
                "split": row["split"],
                "class_id": int(
                    row["class_id"]
                ),
                "class_name": row["class_name"],
                "status": "error",
                "num_frames": 0,
                "valid_ratio": 0.0,
                "top_valid_ratio": 0.0,
                "bottom_valid_ratio": 0.0,
                "joint_path": str(joint_path),
                "score_path": str(score_path),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            print(
                "\nLỗi:",
                clip_id,
                error,
            )

            # Dừng nếu môi trường GPU gặp lỗi hệ thống.
            if consecutive_errors >= 3:
                save_yolo_status()

                raise RuntimeError(
                    "Đã có 3 video liên tiếp lỗi. "
                    "Dừng để tránh đánh dấu lỗi "
                    "toàn bộ dataset."
                ) from error

        if (index + 1) % 25 == 0:
            save_yolo_status()


    save_yolo_status()

    elapsed = perf_counter() - start_time

    print("\nHoàn thành extraction.")
    print(
        "Thời gian:",
        round(elapsed / 60, 2),
        "phút",
    )

    # %% Cell 27
    status_df = pd.read_csv(STATUS_PATH)

    print("Trạng thái:")
    print(
        status_df["status"]
        .value_counts()
    )

    print("\nTổng frame thành công:")
    print(
        status_df.loc[
            status_df["status"] == "success",
            "num_frames",
        ].sum()
    )

    print("\nTỉ lệ frame hợp lệ trung bình:")
    print(
        status_df.loc[
            status_df["status"] == "success",
            "valid_ratio",
        ].mean()
    )

    # %% Cell 28
    joint_files = list(
        YOLO_JOINTS_ROOT.rglob(
            "*_joints.npy"
        )
    )

    score_files = list(
        YOLO_SCORES_ROOT.rglob(
            "*_scores.npy"
        )
    )

    print("Số file joints:", len(joint_files))
    print("Số file scores:", len(score_files))

    invalid_files = []

    for joint_path in tqdm(
        joint_files,
        desc="Validating joints",
    ):
        joints = np.load(
            joint_path,
            mmap_mode="r",
        )

        valid = (
            joints.ndim == 4
            and joints.shape[1:] == (2, 17, 2)
            and len(joints) > 0
        )

        if not valid:
            invalid_files.append(
                (
                    str(joint_path),
                    joints.shape,
                )
            )

    print("File joints hợp lệ:", (
        len(joint_files)
        - len(invalid_files)
    ))
    print("File joints không hợp lệ:", len(invalid_files))

    # %% Cell 29
    root_joint_files = list(
        YOLO_JOINTS_ROOT.glob("*_joints.npy")
    )

    root_score_files = list(
        YOLO_SCORES_ROOT.glob("*_scores.npy")
    )

    print("Joint thử nghiệm:", root_joint_files)
    print("Score thử nghiệm:", root_score_files)

    # %% Cell 30
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt


    COCO_EDGES = [
        (0, 1), (0, 2),
        (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]


    def draw_skeleton(
        frame,
        joints,
        scores,
        score_threshold=0.25,
    ):
        output = frame.copy()

        # Màu BGR của OpenCV.
        player_colors = [
            (255, 255, 0),   # Top: cyan
            (0, 140, 255),   # Bottom: orange
        ]

        player_names = [
            "Top",
            "Bottom",
        ]

        for player_index in range(2):
            color = player_colors[player_index]

            player_joints = joints[player_index]
            player_scores = scores[player_index]

            valid = (
                (player_scores >= score_threshold)
                & np.any(
                    player_joints != 0,
                    axis=1,
                )
            )

            # Vẽ các xương.
            for start, end in COCO_EDGES:
                if valid[start] and valid[end]:
                    start_point = tuple(
                        player_joints[start]
                        .round()
                        .astype(int)
                    )

                    end_point = tuple(
                        player_joints[end]
                        .round()
                        .astype(int)
                    )

                    cv2.line(
                        output,
                        start_point,
                        end_point,
                        color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            # Vẽ các joint.
            for joint_index, point in enumerate(
                player_joints
            ):
                if not valid[joint_index]:
                    continue

                point = tuple(
                    point.round().astype(int)
                )

                cv2.circle(
                    output,
                    point,
                    radius=4,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            # Viết Top hoặc Bottom gần joint hợp lệ đầu tiên.
            valid_indices = np.flatnonzero(valid)

            if len(valid_indices) > 0:
                first_point = (
                    player_joints[valid_indices[0]]
                    .round()
                    .astype(int)
                )

                cv2.putText(
                    output,
                    player_names[player_index],
                    (
                        int(first_point[0]),
                        max(20, int(first_point[1]) - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return output

    # %% Cell 31
    import pandas as pd

    status_df = pd.read_csv(STATUS_PATH)

    status_df["clip_id"] = (
        status_df["clip_id"].astype(str)
    )

    manifest_for_visualization = (
        manifest_df.copy()
    )

    manifest_for_visualization["clip_id"] = (
        manifest_for_visualization[
            "clip_id"
        ].astype(str)
    )

    visualization_df = (
        status_df.loc[
            status_df["status"] == "success"
        ]
        .merge(
            manifest_for_visualization[
                [
                    "clip_id",
                    "video_path",
                ]
            ],
            on="clip_id",
            how="left",
        )
    )

    print(
        "Số video có thể visualize:",
        len(visualization_df),
    )

    display(
        visualization_df[
            [
                "clip_id",
                "class_name",
                "split",
                "num_frames",
                "valid_ratio",
            ]
        ].head()
    )

    # %% Cell 32
    def visualize_video_samples(
        row,
        number_of_frames=6,
        score_threshold=0.25,
    ):
        video_path = str(row["video_path"])
        joint_path = str(row["joint_path"])
        score_path = str(row["score_path"])

        joints = np.load(joint_path)
        scores = np.load(score_path)

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Không mở được video: {video_path}"
            )

        total_frames = min(
            len(joints),
            int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            ),
        )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            num=min(
                number_of_frames,
                total_frames,
            ),
            dtype=int,
        )

        figure, axes = plt.subplots(
            2,
            3,
            figsize=(18, 9),
        )

        axes = np.asarray(axes).reshape(-1)

        for axis_index, frame_index in enumerate(
            frame_indices
        ):
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            opened, frame = capture.read()

            if not opened:
                axes[axis_index].axis("off")
                continue

            annotated = draw_skeleton(
                frame=frame,
                joints=joints[frame_index],
                scores=scores[frame_index],
                score_threshold=score_threshold,
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            )

            axes[axis_index].imshow(
                annotated_rgb
            )

            axes[axis_index].set_title(
                f"Frame {frame_index}"
            )

            axes[axis_index].axis("off")

        for axis in axes[len(frame_indices):]:
            axis.axis("off")

        capture.release()

        figure.suptitle(
            (
                f"{row['class_name']} | "
                f"{row['split']} | "
                f"{row['clip_id']}"
            ),
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()

    # %% Cell 33
    example_row = (
        visualization_df.sample(
            n=1,
            random_state=42,
        )
        .iloc[0]
    )

    visualize_video_samples(
        example_row,
        number_of_frames=6,
    )

    # %% Cell 34
    class_examples = (
        visualization_df
        .groupby(
            "class_name",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=42,
        )
        .sample(
            n=min(
                3,
                visualization_df[
                    "class_name"
                ].nunique(),
            ),
            random_state=42,
        )
    )

    display(
        class_examples[
            [
                "class_name",
                "clip_id",
                "split",
                "valid_ratio",
            ]
        ]
    )

    # %% Cell 35
    for _, example_row in (
        class_examples.iterrows()
    ):
        visualize_video_samples(
            example_row,
            number_of_frames=6,
        )

    # %% Cell 36
    low_quality_examples = (
        visualization_df
        .sort_values("valid_ratio")
        .head(3)
    )

    display(
        low_quality_examples[
            [
                "class_name",
                "clip_id",
                "num_frames",
                "valid_ratio",
                "top_valid_ratio",
                "bottom_valid_ratio",
            ]
        ]
    )

    # %% Cell 37
    for _, example_row in (
        low_quality_examples.iterrows()
    ):
        visualize_video_samples(
            example_row,
            number_of_frames=6,
        )

    # %% Cell 38
    successful_df = status_df.loc[
        status_df["status"] == "success"
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].hist(
        successful_df["valid_ratio"],
        bins=20,
        edgecolor="black",
    )

    axes[0].set_title(
        "Tỉ lệ frame có ít nhất một người"
    )
    axes[0].set_xlabel("Valid ratio")
    axes[0].set_ylabel("Số video")

    axes[1].hist(
        successful_df["top_valid_ratio"],
        bins=20,
        edgecolor="black",
        color="cyan",
    )

    axes[1].set_title(
        "Tỉ lệ phát hiện người Top"
    )
    axes[1].set_xlabel("Top valid ratio")

    axes[2].hist(
        successful_df[
            "bottom_valid_ratio"
        ],
        bins=20,
        edgecolor="black",
        color="orange",
    )

    axes[2].set_title(
        "Tỉ lệ phát hiện người Bottom"
    )
    axes[2].set_xlabel(
        "Bottom valid ratio"
    )

    plt.tight_layout()
    plt.show()
