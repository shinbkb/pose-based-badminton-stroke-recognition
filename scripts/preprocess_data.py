"""Exported from notebooks/data-preprocessing.ipynb.

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
    from zipfile import ZipFile

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import GroupShuffleSplit


    from badminton_stroke.paths import PROJECT_ROOT

    ARCHIVE_PATH = PROJECT_ROOT / "data.zip"
    DATA_DIR = PROJECT_ROOT / "data"
    WORK_ROOT = PROJECT_ROOT / "processed"
    MANIFEST_DIR = WORK_ROOT / "manifests"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    RANDOM_SEED = 42

    print("Project:", PROJECT_ROOT)
    print("Archive:", ARCHIVE_PATH)
    print("Data:", DATA_DIR)
    print("Processed:", WORK_ROOT)

    assert PROJECT_ROOT.exists(), PROJECT_ROOT

    # %% [markdown]
    # ## Tải `data.zip` từ Google Drive
    # 
    # File phải được chia sẻ ở chế độ **Anyone with the link**. Dán link chia sẻ riêng của file `data.zip` vào cell bên dưới.

    # %% Cell 3
    from zipfile import is_zipfile

    import gdown



    GOOGLE_DRIVE_DATA_URL = (
        "PASTE_DATA_ZIP_SHARE_LINK_HERE"
    )

    if ARCHIVE_PATH.exists() and is_zipfile(ARCHIVE_PATH):
        print("ZIP đã tồn tại và hợp lệ:", ARCHIVE_PATH)

    else:
        assert (
            "PASTE_DATA_ZIP_SHARE_LINK_HERE"
            not in GOOGLE_DRIVE_DATA_URL
        ), (
            "Hãy dán link chia sẻ của file data.zip "
            "vào GOOGLE_DRIVE_DATA_URL."
        )

        temporary_archive = (
            ARCHIVE_PATH.parent
            / f"{ARCHIVE_PATH.name}.part"
        )

        print("Đang tải data.zip từ Google Drive...")

        downloaded_path = gdown.download(
            url=GOOGLE_DRIVE_DATA_URL,
            output=str(temporary_archive),
            quiet=False,
            fuzzy=True,
            resume=True,
        )

        assert downloaded_path is not None, (
            "Google Drive không cho tải file. "
            "Hãy kiểm tra quyền Anyone with the link."
        )

        assert is_zipfile(temporary_archive), (
            f"File tải về không phải ZIP hợp lệ: "
            f"{temporary_archive}"
        )

        temporary_archive.replace(
            ARCHIVE_PATH
        )

        print("Đã tải xong:", ARCHIVE_PATH)

    print(
        "Dung lượng ZIP:",
        f"{ARCHIVE_PATH.stat().st_size / (1024 ** 3):.2f} GB",
    )

    # %% Cell 4
    dataset_candidates = [
        path
        for path in DATA_DIR.rglob(
            "VideoBadminton_Dataset"
        )
        if path.is_dir()
    ]

    if not dataset_candidates:
        print("Đang giải nén data.zip...")

        with ZipFile(ARCHIVE_PATH, "r") as archive:
            archive.extractall(DATA_DIR)

        dataset_candidates = [
            path
            for path in DATA_DIR.rglob(
                "VideoBadminton_Dataset"
            )
            if path.is_dir()
        ]

    print(
        "Số thư mục VideoBadminton_Dataset:",
        len(dataset_candidates),
    )

    for candidate in dataset_candidates:
        print("-", candidate)

    # %% Cell 5
    assert len(dataset_candidates) == 1, (
        "Cần tìm thấy đúng một thư mục "
        f"VideoBadminton_Dataset: {dataset_candidates}"
    )

    DATA_ROOT = dataset_candidates[0]

    print("DATA_ROOT:", DATA_ROOT)

    # %% [markdown]
    # ### Kiểm tra các lớp và video

    # %% Cell 7
    class_dirs = sorted(
        path
        for path in DATA_ROOT.iterdir()
        if path.is_dir()
    )

    video_paths = sorted(
        DATA_ROOT.rglob("*.mp4")
    )

    print("Số lớp:", len(class_dirs))
    print("Số video:", len(video_paths))

    for class_dir in class_dirs:
        num_videos = len(
            list(class_dir.glob("*.mp4"))
        )

        print(
            f"{class_dir.name:35s}: "
            f"{num_videos}"
        )

    assert len(class_dirs) == 18
    assert len(video_paths) > 0

    # %% [markdown]
    # ## Tạo danh sách lớp

    # %% Cell 9
    class_records = []

    for class_dir in class_dirs:
        class_id_text, class_name = (
            class_dir.name.split(
                "_",
                maxsplit=1,
            )
        )

        class_records.append({
            "class_id": int(class_id_text),
            "class_name": class_name,
            "class_dir": class_dir.name,
        })

    classes_df = (
        pd.DataFrame(class_records)
        .sort_values("class_id")
        .reset_index(drop=True)
    )

    assert (
        classes_df["class_id"].tolist()
        == list(range(18))
    )

    classes_df

    # %% [markdown]
    # ### Đọc thông tin từ tên video

    # %% Cell 11
    import re


    FILENAME_PATTERN = re.compile(
        r"^"
        r"(?P<date>\d{4}-\d{2}-\d{2})_"
        r"(?P<time>\d{2}-\d{2}-\d{2})_"
        r"dataset_"
        r"(?P<set_name>set\d+)_"
        r"(?P<stroke_index>\d+)_"
        r"(?P<start_frame>\d+)_"
        r"(?P<end_frame>\d+)_"
        r"(?P<player>[A-Za-z]+)_"
        r"(?P<filename_class_id>\d+)"
        r"$"
    )


    def parse_video_path(video_path: Path) -> dict:
        match = FILENAME_PATTERN.match(
            video_path.stem
        )

        if match is None:
            raise ValueError(
                f"Tên file không hợp lệ: "
                f"{video_path.name}"
            )

        metadata = match.groupdict()

        folder_class_id, folder_class_name = (
            video_path.parent.name.split(
                "_",
                maxsplit=1,
            )
        )

        start_frame = int(
            metadata["start_frame"]
        )

        end_frame = int(
            metadata["end_frame"]
        )

        return {
            "clip_id": video_path.stem,
            "filename": video_path.name,

            # Dùng relative_path để không phụ thuộc
            # đường dẫn Colab hoặc Kaggle.
            "relative_path": str(
                video_path.relative_to(DATA_ROOT)
            ),

            "video_path": str(video_path),

            # Các clip cùng phiên quay phải nằm
            # trong cùng một split.
            "source_id": (
                f"{metadata['date']}_"
                f"{metadata['time']}"
            ),

            "date": metadata["date"],
            "time": metadata["time"],
            "set_name": metadata["set_name"],

            "stroke_index": int(
                metadata["stroke_index"]
            ),

            "start_frame": start_frame,
            "end_frame": end_frame,

            "expected_frames": (
                end_frame - start_frame + 1
            ),

            "player": metadata["player"],

            "filename_class_id": int(
                metadata["filename_class_id"]
            ),

            "class_id": int(folder_class_id),
            "class_name": folder_class_name,

            "file_size_bytes": (
                video_path.stat().st_size
            ),
        }

    # %% Cell 12
    records = []
    parse_errors = []

    for video_path in video_paths:
        try:
            records.append(
                parse_video_path(video_path)
            )

        except Exception as error:
            parse_errors.append({
                "video_path": str(video_path),
                "error": str(error),
            })

    manifest_df = pd.DataFrame(records)
    parse_errors_df = pd.DataFrame(parse_errors)

    print("Tổng video:", len(video_paths))
    print("Đọc thành công:", len(manifest_df))
    print("Lỗi tên file:", len(parse_errors_df))

    manifest_df.head()

    # %% [markdown]
    # ## Kiểm tra và lọc dữ liệu lỗi

    # %% Cell 14
    manifest_df["has_valid_frame_range"] = (
        manifest_df["expected_frames"] > 0
    )

    manifest_df["is_non_empty"] = (
        manifest_df["file_size_bytes"] > 0
    )

    manifest_df["has_standard_player_code"] = (
        manifest_df["player"].isin(["A", "B"])
    )

    manifest_df["class_id_matches_filename"] = (
        manifest_df["class_id"]
        == manifest_df["filename_class_id"]
    )

    print(
        "Frame range không hợp lệ:",
        (~manifest_df[
            "has_valid_frame_range"
        ]).sum(),
    )

    print(
        "File rỗng:",
        (~manifest_df[
            "is_non_empty"
        ]).sum(),
    )

    print(
        "Player khác A/B:",
        (~manifest_df[
            "has_standard_player_code"
        ]).sum(),
    )

    print(
        "Class ID không khớp:",
        (~manifest_df[
            "class_id_matches_filename"
        ]).sum(),
    )
    manifest_df["is_usable"] = (
        manifest_df["has_valid_frame_range"]
        & manifest_df["is_non_empty"]
    )

    clean_df = (
        manifest_df[
            manifest_df["is_usable"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("Tổng clip:", len(manifest_df))
    print("Clip sử dụng được:", len(clean_df))
    print(
        "Clip bị loại:",
        len(manifest_df) - len(clean_df),
    )

    # %% [markdown]
    # ## Chia train, validation và test theo phiên quay

    # %% Cell 16
    def class_distribution(
        dataframe,
        class_ids,
    ):
        counts = (
            dataframe["class_id"]
            .value_counts()
            .reindex(
                class_ids,
                fill_value=0,
            )
            .sort_index()
            .to_numpy(dtype=np.float64)
        )

        return counts / max(
            counts.sum(),
            1,
        )


    def choose_group_split(
        dataframe,
        test_size,
        group_column="source_id",
        num_trials=2000,
        random_seed=42,
    ):
        class_ids = sorted(
            dataframe["class_id"].unique()
        )

        overall_distribution = (
            class_distribution(
                dataframe,
                class_ids,
            )
        )

        best_score = float("inf")
        best_train_indices = None
        best_test_indices = None

        for trial in range(num_trials):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=(
                    random_seed + trial
                ),
            )

            train_indices, test_indices = next(
                splitter.split(
                    dataframe,
                    groups=dataframe[
                        group_column
                    ],
                )
            )

            train_part = dataframe.iloc[
                train_indices
            ]

            test_part = dataframe.iloc[
                test_indices
            ]

            size_error = abs(
                len(test_part) / len(dataframe)
                - test_size
            )

            train_distribution = (
                class_distribution(
                    train_part,
                    class_ids,
                )
            )

            test_distribution = (
                class_distribution(
                    test_part,
                    class_ids,
                )
            )

            distribution_error = (
                np.abs(
                    train_distribution
                    - overall_distribution
                ).mean()
                +
                np.abs(
                    test_distribution
                    - overall_distribution
                ).mean()
            )

            score = (
                size_error
                + distribution_error
            )

            if score < best_score:
                best_score = score
                best_train_indices = (
                    train_indices
                )
                best_test_indices = (
                    test_indices
                )

        return (
            dataframe.iloc[
                best_train_indices
            ].copy(),
            dataframe.iloc[
                best_test_indices
            ].copy(),
        )
    train_val_df, test_df = (
        choose_group_split(
            dataframe=clean_df,
            test_size=0.15,
            random_seed=RANDOM_SEED,
        )
    )

    train_df, val_df = (
        choose_group_split(
            dataframe=train_val_df,
            test_size=0.15 / 0.85,
            random_seed=(
                RANDOM_SEED + 10_000
            ),
        )
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat(
        [
            train_df,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("Tổng:", len(split_df))

    # %% [markdown]
    # ## Lưu Manifest

    # %% Cell 18
    manifest_df.to_csv(
        MANIFEST_DIR
        / "all_videos_audit.csv",
        index=False,
    )

    split_df.to_csv(
        MANIFEST_DIR
        / "dataset_manifest.csv",
        index=False,
    )

    train_df.to_csv(
        MANIFEST_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        MANIFEST_DIR / "val.csv",
        index=False,
    )

    test_df.to_csv(
        MANIFEST_DIR / "test.csv",
        index=False,
    )

    classes_df.to_csv(
        MANIFEST_DIR / "classes.csv",
        index=False,
    )

    if not parse_errors_df.empty:
        parse_errors_df.to_csv(
            MANIFEST_DIR
            / "parse_errors.csv",
            index=False,
        )

    print("Đã lưu tại:", MANIFEST_DIR)

    for path in MANIFEST_DIR.iterdir():
        print("-", path.name)

    # %% Cell 19
    saved_manifest = pd.read_csv(
        MANIFEST_DIR
        / "dataset_manifest.csv"
    )

    print(saved_manifest.shape)
    print(
        saved_manifest["split"]
        .value_counts()
    )

    saved_manifest.head()

    # %% [markdown]
    # # Joint Extraction

    # %% Cell 21
    # Kiểm tra GPU và tiến trình đang sử dụng GPU.
    import subprocess; subprocess.run(["nvidia-smi"], check=False)

    # %% Cell 22
    import onnxruntime as ort

    # Nạp CUDA/cuDNN được cài bởi extra [cuda,cudnn].
    ort.preload_dlls(directory="")

    print("ONNX Runtime:", ort.__version__)
    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    assert ort.__version__ == "1.26.0", (
        "Notebook được kiểm thử với "
        "onnxruntime-gpu==1.26.0."
    )

    assert "CUDAExecutionProvider" in (
        ort.get_available_providers()
    ), "ONNX Runtime chưa nhận CUDA GPU."

    # %% [markdown]
    # ## Môi trường server
    # 
    # Môi trường đã được cài bằng `uv sync --frozen`. Không chạy `pip install` trong notebook.

    # %% Cell 24
    from importlib.metadata import version

    import cv2
    import rtmlib

    print("RTMLib:", version("rtmlib"))
    print("OpenCV:", cv2.__version__)

    # %% Cell 25
    # Xác nhận lại provider trước khi tải model.
    print("ONNX Runtime:", ort.__version__)
    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    # %% [markdown]
    # ## Đọc lại manifest và khôi phục đường dẫn video

    # %% Cell 27
    from pathlib import Path

    import cv2
    import numpy as np
    import pandas as pd


    MANIFEST_PATH = (
        MANIFEST_DIR
        / "dataset_manifest.csv"
    )

    assert MANIFEST_PATH.exists(), (
        f"Không tìm thấy: {MANIFEST_PATH}"
    )

    manifest_df = pd.read_csv(
        MANIFEST_PATH
    )

    assert DATA_ROOT.exists(), DATA_ROOT

    manifest_df["video_path"] = (
        manifest_df["relative_path"]
        .apply(
            lambda relative_path: str(
                DATA_ROOT / relative_path
            )
        )
    )

    if "clip_id" not in manifest_df.columns:
        manifest_df["clip_id"] = (
            manifest_df["filename"]
            .apply(
                lambda name: Path(name).stem
            )
        )

    manifest_df = (
        manifest_df
        .sort_values(
            [
                "split",
                "class_id",
                "clip_id",
            ]
        )
        .reset_index(drop=True)
    )

    manifest_df["video_exists"] = (
        manifest_df["video_path"]
        .apply(
            lambda video_path: Path(
                video_path
            ).exists()
        )
    )

    print("Tổng clip:", len(manifest_df))
    print(
        manifest_df["split"].value_counts()
    )
    print(
        "Video bị thiếu:",
        (~manifest_df["video_exists"]).sum(),
    )

    assert manifest_df["video_exists"].all()

    # %% [markdown]
    # ## Khởi tạo RTMPose

    # %% Cell 29
    from importlib.metadata import version

    print("RTMLib:", version("rtmlib"))
    print("ONNX Runtime:", ort.__version__)
    print(
        "Providers:",
        ort.get_available_providers(),
    )

    # %% Cell 30
    from rtmlib import Body, PoseTracker

    pose_model = PoseTracker(
        Body,
        mode="balanced",
        det_frequency=5,
        tracking=False,
        backend="onnxruntime",
        device="cuda",
        to_openpose=False,
    )

    detector_providers = (
        pose_model.det_model
        .session
        .get_providers()
    )

    pose_providers = (
        pose_model.pose_model
        .session
        .get_providers()
    )

    print("Detector:", detector_providers)
    print("Pose:", pose_providers)

    assert detector_providers[0] == (
        "CUDAExecutionProvider"
    )

    assert pose_providers[0] == (
        "CUDAExecutionProvider"
    )

    # %% Cell 31
    def pose_area(
        keypoints,
        scores,
        threshold=0.2,
    ):
        valid = scores >= threshold

        if valid.sum() < 4:
            return 0.0

        valid_points = keypoints[valid]

        width = (
            valid_points[:, 0].max()
            - valid_points[:, 0].min()
        )

        height = (
            valid_points[:, 1].max()
            - valid_points[:, 1].min()
        )

        return float(width * height)


    def vertical_position(
        keypoints,
        scores,
        threshold=0.2,
    ):
        ankle_indices = [15, 16]

        valid_ankles = [
            index
            for index in ankle_indices
            if scores[index] >= threshold
        ]

        if valid_ankles:
            return float(
                keypoints[
                    valid_ankles,
                    1,
                ].mean()
            )

        valid = scores >= threshold

        if valid.any():
            return float(
                keypoints[valid, 1].mean()
            )

        return float("inf")


    def select_two_players(
        keypoints,
        scores,
    ):
        if len(keypoints) < 2:
            return None, None

        areas = [
            pose_area(
                person_keypoints,
                person_scores,
            )
            for person_keypoints, person_scores
            in zip(keypoints, scores)
        ]

        # Chọn hai người có vùng pose lớn nhất.
        selected_indices = np.argsort(
            areas
        )[-2:]

        selected_keypoints = keypoints[
            selected_indices
        ]

        selected_scores = scores[
            selected_indices
        ]

        # Người ở phía trên sân trước,
        # người phía dưới sân sau.
        order = sorted(
            range(2),
            key=lambda index: vertical_position(
                selected_keypoints[index],
                selected_scores[index],
            ),
        )

        return (
            selected_keypoints[order],
            selected_scores[order],
        )

    # %% [markdown]
    # ## Chuẩn hóa joint

    # %% Cell 33
    def normalize_joints(
        keypoints,
        scores,
        threshold=0.2,
    ):
        result = np.zeros(
            (17, 2),
            dtype=np.float32,
        )

        valid = scores >= threshold

        if valid.sum() < 4:
            return result

        valid_points = keypoints[valid]

        min_xy = valid_points.min(axis=0)
        max_xy = valid_points.max(axis=0)

        center = (
            min_xy + max_xy
        ) / 2

        diagonal = np.linalg.norm(
            max_xy - min_xy
        )

        if diagonal <= 0:
            return result

        normalized = (
            keypoints - center
        ) / diagonal

        normalized[~valid] = 0.0

        number_of_joints = min(
            17,
            len(normalized),
        )

        result[:number_of_joints] = (
            normalized[:number_of_joints]
        )

        return result

    # %% Cell 34
    def extract_video_joints(
        video_path,
        pose_model,
    ):
        # Xóa trạng thái từ video trước.
        pose_model.reset()

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Không mở được video: "
                f"{video_path}"
            )

        frames_joints = []

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                keypoints, scores = (
                    pose_model(frame)
                )

                keypoints = np.asarray(
                    keypoints,
                    dtype=np.float32,
                )

                scores = np.asarray(
                    scores,
                    dtype=np.float32,
                )

                (
                    selected_keypoints,
                    selected_scores,
                ) = select_two_players(
                    keypoints,
                    scores,
                )

                if selected_keypoints is None:
                    frame_joints = np.zeros(
                        (2, 17, 2),
                        dtype=np.float32,
                    )

                else:
                    frame_joints = np.stack(
                        [
                            normalize_joints(
                                person_keypoints,
                                person_scores,
                            )
                            for (
                                person_keypoints,
                                person_scores,
                            )
                            in zip(
                                selected_keypoints,
                                selected_scores,
                            )
                        ],
                        axis=0,
                    )

                frames_joints.append(
                    frame_joints
                )

        finally:
            capture.release()

        if not frames_joints:
            return np.zeros(
                (0, 2, 17, 2),
                dtype=np.float32,
            )

        return np.stack(
            frames_joints,
            axis=0,
        )

    # %% [markdown]
    # ## Benchmark trước khi chạy toàn bộ

    # %% Cell 36
    import time


    sample_row = manifest_df.iloc[0]

    for run_index in range(2):
        start_time = time.time()

        sample_joints = (
            extract_video_joints(
                sample_row["video_path"],
                pose_model,
            )
        )

        elapsed = time.time() - start_time

        fps = (
            len(sample_joints) / elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"Run {run_index + 1}: "
            f"{elapsed:.2f} giây, "
            f"{fps:.2f} FPS, "
            f"shape={sample_joints.shape}"
        )

    # %% Cell 37
    JOINTS_ROOT = (
        WORK_ROOT
        / "joints_npy_balanced"
    )

    STATUS_PATH = (
        WORK_ROOT
        / "joint_extraction_status.csv"
    )

    JOINTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Joints:", JOINTS_ROOT)
    print("Status:", STATUS_PATH)

    # %% [markdown]
    # ## Chạy toàn bộ dataset

    # %% Cell 39
    from tqdm import tqdm


    if STATUS_PATH.exists():
        old_status_df = pd.read_csv(
            STATUS_PATH
        )

        status_records = {
            row["clip_id"]: row
            for row in old_status_df.to_dict(
                orient="records"
            )
        }
    else:
        status_records = {}


    def get_output_path(row):
        class_directory = (
            f"{int(row.class_id):02d}_"
            f"{row.class_name}"
        )

        output_directory = (
            JOINTS_ROOT
            / row.split
            / class_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            output_directory
            / f"{row.clip_id}_joints.npy"
        )


    def save_status():
        if status_records:
            pd.DataFrame(
                status_records.values()
            ).to_csv(
                STATUS_PATH,
                index=False,
            )

    # %% Cell 40
    for index, row in enumerate(
        tqdm(
            manifest_df.itertuples(
                index=False
            ),
            total=len(manifest_df),
            desc="Extracting joints",
        )
    ):
        output_path = get_output_path(row)

        # Tiếp tục an toàn sau khi SSH hoặc notebook bị ngắt.
        if output_path.exists():
            if row.clip_id not in status_records:
                existing_joints = np.load(
                    output_path
                )

                valid_frames = np.any(
                    existing_joints != 0,
                    axis=(1, 2, 3),
                )

                status_records[row.clip_id] = {
                    "clip_id": row.clip_id,
                    "split": row.split,
                    "class_id": int(row.class_id),
                    "class_name": row.class_name,
                    "status": "success",
                    "num_frames": len(
                        existing_joints
                    ),
                    "valid_ratio": float(
                        valid_frames.mean()
                    ),
                    "output_path": str(
                        output_path
                    ),
                    "error": "",
                }

            continue

        try:
            joints = extract_video_joints(
                row.video_path,
                pose_model,
            )

            if len(joints) == 0:
                raise RuntimeError(
                    "Video không có frame."
                )

            temporary_path = (
                output_path.parent
                / (
                    output_path.stem
                    + ".part.npy"
                )
            )

            np.save(
                temporary_path,
                joints.astype(np.float32),
            )

            temporary_path.replace(
                output_path
            )

            valid_frames = np.any(
                joints != 0,
                axis=(1, 2, 3),
            )

            status_records[row.clip_id] = {
                "clip_id": row.clip_id,
                "split": row.split,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "status": "success",
                "num_frames": len(joints),
                "valid_ratio": float(
                    valid_frames.mean()
                ),
                "output_path": str(
                    output_path
                ),
                "error": "",
            }

        except Exception as error:
            status_records[row.clip_id] = {
                "clip_id": row.clip_id,
                "split": row.split,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "status": "error",
                "num_frames": 0,
                "valid_ratio": 0.0,
                "output_path": str(
                    output_path
                ),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            print(
                "\nLỗi:",
                row.clip_id,
                error,
            )

        if (index + 1) % 100 == 0:
            save_status()


    save_status()

    joint_files = list(
        JOINTS_ROOT.rglob(
            "*_joints.npy"
        )
    )

    status_df = pd.DataFrame(
        status_records.values()
    )

    print("Hoàn thành extraction.")
    print("Số file joint:", len(joint_files))
    print("Trạng thái:")
    print(status_df["status"].value_counts())

    # %% Cell 41
    from pathlib import Path
    import numpy as np

    JOINTS_ROOT = Path(
        "/home/dutai/khacbinh/pbl5_doantotnghiep/"
        "processed/joints_npy_balanced"
    )

    joint_files = list(JOINTS_ROOT.rglob("*_joints.npy"))

    print("Số file:", len(joint_files))

    invalid_files = []

    for path in joint_files:
        try:
            joints = np.load(path, mmap_mode="r")

            # Dạng mong đợi: (số frame, 2 người, 17 joint, 2 tọa độ)
            valid = (
                joints.ndim == 4
                and joints.shape[1:] == (2, 17, 2)
                and joints.shape[0] > 0
            )

            if not valid:
                invalid_files.append((str(path), joints.shape))

        except Exception as error:
            invalid_files.append((str(path), str(error)))

    print("File hợp lệ:", len(joint_files) - len(invalid_files))
    print("File không hợp lệ:", len(invalid_files))

    for item in invalid_files[:10]:
        print(item)
