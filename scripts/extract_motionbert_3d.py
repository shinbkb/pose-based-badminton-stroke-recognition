"""Exported from notebooks/motionbert_3d_extraction.ipynb.

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
    import os
    import sys
    import zipfile

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from tqdm import tqdm

    # %% Cell 2
    from badminton_stroke.paths import PROJECT_ROOT

    MOTIONBERT_ROOT = (
        PROJECT_ROOT
        / "third_party"
        / "MotionBERT"
    )

    MOTIONBERT_CONFIG = (
        MOTIONBERT_ROOT
        / "configs"
        / "pose3d"
        / "MB_ft_h36m_global_lite.yaml"
    )

    MOTIONBERT_CHECKPOINT_ROOT = (
        MOTIONBERT_ROOT
        / "checkpoint"
        / "pose3d"
        / "FT_MB_lite_MB_ft_h36m_global_lite"
    )

    JOINTS_2D_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "joints_npy_yolo26s_pose"
    )

    SCORES_2D_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "scores_npy_yolo26s_pose"
    )

    JOINTS_3D_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "joints_3d_motionbert"
    )

    SCORES_H36M_ROOT = (
        PROJECT_ROOT
        / "processed"
        / "scores_h36m_motionbert"
    )

    CHECKPOINT_CACHE_ROOT = (
        PROJECT_ROOT
        / "processed"
        / ".motionbert_checkpoint_cache"
    )

    required_paths = {
        "MotionBERT": MOTIONBERT_ROOT,
        "MotionBERT config": MOTIONBERT_CONFIG,
        "MotionBERT checkpoint directory": MOTIONBERT_CHECKPOINT_ROOT,
        "2D joints": JOINTS_2D_ROOT,
        "2D scores": SCORES_2D_ROOT,
    }

    for path_name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path_name} không tồn tại: {path}"
            )

    JOINTS_3D_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SCORES_H36M_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)

    # %% Cell 3
    if str(MOTIONBERT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(MOTIONBERT_ROOT),
        )

    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone
    from lib.utils.utils_data import crop_scale

    # %% Cell 4
    def is_git_lfs_pointer(path):
        if not path.is_file() or path.stat().st_size > 1024:
            return False

        try:
            header = path.read_bytes()[:200]
        except OSError:
            return False

        return b"git-lfs.github.com/spec" in header


    def repair_wrapped_torch_checkpoint(path):
        """Chuẩn hóa ZIP có thêm thư mục bọc, giữ nguyên checkpoint gốc."""
        if not zipfile.is_zipfile(path):
            return path

        with zipfile.ZipFile(path) as source_zip:
            names = source_zip.namelist()
            pickle_names = [name for name in names if name.endswith("/data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError(f"Checkpoint phải chứa đúng một data.pkl: {path}")

            prefix = pickle_names[0].rsplit("/", 1)[0] + "/"
            if prefix + "version" not in names:
                raise ValueError(f"Checkpoint thiếu bản ghi version: {path}")

            if len(prefix.rstrip("/").split("/")) == 1:
                return path

            output_path = CHECKPOINT_CACHE_ROOT / (path.name + "_normalized.pt")
            if (output_path.is_file()
                    and output_path.stat().st_mtime >= path.stat().st_mtime):
                return output_path

            temporary_path = output_path.with_suffix(".tmp")
            with zipfile.ZipFile(
                temporary_path, "w", compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as target_zip:
                for name in names:
                    if name.startswith(prefix) and not name.endswith("/"):
                        target_zip.writestr(
                            "archive/" + name[len(prefix):], source_zip.read(name)
                        )
            temporary_path.replace(output_path)

        print("Đã chuẩn hóa cấu trúc ZIP checkpoint:", output_path)
        return output_path


    def repack_extracted_torch_checkpoint(
        data_pickle,
    ):
        """Đóng gói lại checkpoint PyTorch đã bị giải nén thành archive/."""
        archive_directory = data_pickle.parent

        if archive_directory.name != "archive":
            raise ValueError(
                f"Cấu trúc checkpoint không hợp lệ: {data_pickle}"
            )

        source_directory = archive_directory.parent
        cache_name = (
            source_directory.name.replace(".", "_")
            + "_repacked.pt"
        )
        output_path = CHECKPOINT_CACHE_ROOT / cache_name

        source_files = [
            path
            for path in archive_directory.rglob("*")
            if path.is_file()
        ]

        if not source_files:
            raise FileNotFoundError(
                f"Checkpoint giải nén không có dữ liệu: {archive_directory}"
            )

        newest_source_time = max(
            path.stat().st_mtime
            for path in source_files
        )

        if (
            output_path.is_file()
            and output_path.stat().st_mtime >= newest_source_time
        ):
            return output_path

        temporary_path = output_path.with_suffix(".tmp")

        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as checkpoint_zip:
            for source_path in source_files:
                archive_name = source_path.relative_to(
                    archive_directory.parent
                )
                checkpoint_zip.write(
                    source_path,
                    arcname=str(archive_name),
                )

        temporary_path.replace(output_path)
        return output_path


    def resolve_motionbert_checkpoint(
        checkpoint_root,
    ):
        """Tìm file checkpoint hoặc sửa cấu trúc checkpoint bị giải nén."""
        checkpoint_root = Path(checkpoint_root)

        preferred_names = (
            "best_epoch.bin",
            "latest_epoch.bin",
            "best_epoch.pt",
            "latest_epoch.pt",
            "best_epoch",
            "latest_epoch",
        )

        for file_name in preferred_names:
            candidate = checkpoint_root / file_name
            if candidate.is_file():
                if is_git_lfs_pointer(candidate):
                    raise RuntimeError(
                        f"{candidate} chỉ là Git LFS pointer. "
                        "Hãy tải file checkpoint thật bằng git lfs pull."
                    )
                return repair_wrapped_torch_checkpoint(candidate)

        checkpoint_files = sorted(
            (
                path
                for path in checkpoint_root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in {".bin", ".pt", ".pth", ".ckpt"}
                )
            ),
            key=lambda path: (
                "best" not in path.name.lower(),
                -path.stat().st_size,
            ),
        )

        for candidate in checkpoint_files:
            if not is_git_lfs_pointer(candidate):
                return repair_wrapped_torch_checkpoint(candidate)

        extracted_archives = sorted(
            checkpoint_root.rglob("archive/data.pkl"),
            key=lambda path: (
                "best" not in str(path).lower(),
                str(path),
            ),
        )

        if extracted_archives:
            repaired_path = repack_extracted_torch_checkpoint(
                extracted_archives[0]
            )
            print(
                "Checkpoint đã bị giải nén; đã đóng gói lại:",
                repaired_path,
            )
            return repaired_path

        raise FileNotFoundError(
            "Không tìm thấy checkpoint MotionBERT hợp lệ trong "
            f"{checkpoint_root}. Cần file best_epoch.bin/latest_epoch.bin "
            "hoặc thư mục chứa archive/data.pkl."
        )


    MOTIONBERT_CHECKPOINT = resolve_motionbert_checkpoint(
        MOTIONBERT_CHECKPOINT_ROOT
    )

    print("Checkpoint:", MOTIONBERT_CHECKPOINT)

    motionbert_config = get_config(
        str(MOTIONBERT_CONFIG)
    )

    motionbert = load_backbone(
        motionbert_config
    )

    checkpoint = torch.load(
        str(MOTIONBERT_CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint MotionBERT phải là dictionary, "
            f"nhận được {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get(
        "model_pos",
        checkpoint.get(
            "state_dict",
            checkpoint.get("model"),
        ),
    )

    if state_dict is None:
        tensor_values = all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )

        if tensor_values:
            state_dict = checkpoint
        else:
            raise KeyError(
                "Checkpoint không có model_pos, state_dict hoặc model. "
                f"Các key hiện có: {list(checkpoint)[:20]}"
            )

    # Loại tiền tố module. nếu checkpoint được lưu
    # bằng DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    motionbert.load_state_dict(
        state_dict,
        strict=True,
    )

    motionbert = motionbert.to(DEVICE)
    motionbert.eval()

    print(
        "MotionBERT parameters:",
        f"{sum(p.numel() for p in motionbert.parameters()):,}",
    )

    # %% Cell 5
    def merge_two_points(
        xy,
        scores,
        first,
        second,
    ):
        first_xy = xy[..., first, :]
        second_xy = xy[..., second, :]

        first_score = scores[..., first]
        second_score = scores[..., second]

        first_valid = first_score > 0
        second_valid = second_score > 0

        result_xy = np.zeros_like(
            first_xy,
            dtype=np.float32,
        )

        result_score = np.zeros_like(
            first_score,
            dtype=np.float32,
        )

        both = first_valid & second_valid
        only_first = first_valid & ~second_valid
        only_second = second_valid & ~first_valid

        result_xy[both] = (
            first_xy[both]
            + second_xy[both]
        ) / 2.0

        result_score[both] = np.minimum(
            first_score[both],
            second_score[both],
        )

        result_xy[only_first] = (
            first_xy[only_first]
        )

        result_score[only_first] = (
            first_score[only_first] * 0.5
        )

        result_xy[only_second] = (
            second_xy[only_second]
        )

        result_score[only_second] = (
            second_score[only_second] * 0.5
        )

        return result_xy, result_score

    # %% Cell 6
    def coco17_to_h36m17(
        coco_xy,
        coco_scores,
    ):
        """
        coco_xy:
            (T, M, 17, 2)

        coco_scores:
            (T, M, 17)

        return:
            h36m: (T, M, 17, 3)

        Kênh cuối ở đây vẫn là:
            x, y, confidence
        """

        num_frames, num_people = (
            coco_xy.shape[:2]
        )

        h36m = np.zeros(
            (
                num_frames,
                num_people,
                17,
                3,
            ),
            dtype=np.float32,
        )

        pelvis_xy, pelvis_score = (
            merge_two_points(
                coco_xy,
                coco_scores,
                11,
                12,
            )
        )

        neck_xy, neck_score = (
            merge_two_points(
                coco_xy,
                coco_scores,
                5,
                6,
            )
        )

        eye_xy, eye_score = (
            merge_two_points(
                coco_xy,
                coco_scores,
                1,
                2,
            )
        )

        # Ước lượng đỉnh đầu từ hướng neck -> eyes.
        head_xy = (
            eye_xy
            + 0.5 * (
                eye_xy - neck_xy
            )
        )

        head_score = np.minimum(
            eye_score,
            neck_score,
        )

        spine_xy = (
            pelvis_xy + neck_xy
        ) / 2.0

        spine_score = np.minimum(
            pelvis_score,
            neck_score,
        )

        def assign(
            h36m_index,
            source_xy,
            source_score,
        ):
            h36m[
                ..., h36m_index, :2
            ] = source_xy

            h36m[
                ..., h36m_index, 2
            ] = source_score

        # Pelvis.
        assign(
            0,
            pelvis_xy,
            pelvis_score,
        )

        # Chân phải.
        assign(1, coco_xy[..., 12, :],
               coco_scores[..., 12])
        assign(2, coco_xy[..., 14, :],
               coco_scores[..., 14])
        assign(3, coco_xy[..., 16, :],
               coco_scores[..., 16])

        # Chân trái.
        assign(4, coco_xy[..., 11, :],
               coco_scores[..., 11])
        assign(5, coco_xy[..., 13, :],
               coco_scores[..., 13])
        assign(6, coco_xy[..., 15, :],
               coco_scores[..., 15])

        # Thân và đầu.
        assign(7, spine_xy, spine_score)
        assign(8, neck_xy, neck_score)
        assign(9, coco_xy[..., 0, :],
               coco_scores[..., 0])
        assign(10, head_xy, head_score)

        # Tay trái.
        assign(11, coco_xy[..., 5, :],
               coco_scores[..., 5])
        assign(12, coco_xy[..., 7, :],
               coco_scores[..., 7])
        assign(13, coco_xy[..., 9, :],
               coco_scores[..., 9])

        # Tay phải.
        assign(14, coco_xy[..., 6, :],
               coco_scores[..., 6])
        assign(15, coco_xy[..., 8, :],
               coco_scores[..., 8])
        assign(16, coco_xy[..., 10, :],
               coco_scores[..., 10])

        return h36m

    # %% Cell 7
    def interpolate_missing_keypoints(
        motion_2d,
    ):
        """
        motion_2d: (T, M, 17, 3)
                   x, y, confidence
        """

        result = motion_2d.copy()

        num_frames, num_people, num_joints = (
            result.shape[:3]
        )

        frame_indices = np.arange(
            num_frames
        )

        for person_index in range(
            num_people
        ):
            for joint_index in range(
                num_joints
            ):
                confidence = result[
                    :,
                    person_index,
                    joint_index,
                    2,
                ]

                valid = confidence > 0

                if valid.sum() == 0:
                    continue

                for coordinate in [0, 1]:
                    values = result[
                        :,
                        person_index,
                        joint_index,
                        coordinate,
                    ]

                    result[
                        :,
                        person_index,
                        joint_index,
                        coordinate,
                    ] = np.interp(
                        frame_indices,
                        frame_indices[valid],
                        values[valid],
                    )

        return result

    # %% Cell 8
    def normalize_motionbert_input(
        h36m_2d,
    ):
        """
        h36m_2d: (T, M, 17, 3)
        """

        output = np.zeros_like(
            h36m_2d,
            dtype=np.float32,
        )

        num_people = h36m_2d.shape[1]

        for person_index in range(
            num_people
        ):
            person_motion = h36m_2d[
                :, person_index
            ]

            output[
                :, person_index
            ] = crop_scale(
                person_motion,
                scale_range=[1, 1],
            )

        return output

    # %% Cell 9
    H36M_LEFT = [
        4, 5, 6,
        11, 12, 13,
    ]

    H36M_RIGHT = [
        1, 2, 3,
        14, 15, 16,
    ]


    def flip_h36m(
        tensor,
    ):
        """
        tensor:
            (N, T, 17, C)
        """

        flipped = tensor.clone()

        flipped[..., 0] *= -1

        indices = (
            H36M_LEFT
            + H36M_RIGHT
        )

        swapped = (
            H36M_RIGHT
            + H36M_LEFT
        )

        flipped[
            ...,
            indices,
            :,
        ] = flipped[
            ...,
            swapped,
            :,
        ]

        return flipped

    # %% Cell 10
    @torch.inference_mode()
    def lift_yolo_pose_to_3d(
        joints_2d,
        scores_2d,
    ):
        """
        joints_2d:
            (T, 2, 17, 2), COCO

        scores_2d:
            (T, 2, 17)

        return:
            joints_3d:
                (T, 2, 17, 3), H36M

            h36m_scores:
                (T, 2, 17)
        """

        assert joints_2d.ndim == 4
        assert joints_2d.shape[1:] == (
            2, 17, 2
        )

        assert scores_2d.shape == (
            *joints_2d.shape[:-1],
        )

        num_frames = len(joints_2d)

        assert num_frames <= 243, (
            "MotionBERT checkpoint này hỗ trợ "
            "tối đa 243 frame."
        )

        h36m_2d = coco17_to_h36m17(
            joints_2d,
            scores_2d,
        )

        h36m_2d = (
            interpolate_missing_keypoints(
                h36m_2d
            )
        )

        h36m_scores = h36m_2d[..., 2].copy()

        normalized = (
            normalize_motionbert_input(
                h36m_2d
            )
        )

        # (T, M, J, C)
        # -> (M, T, J, C)
        model_input = (
            normalized
            .transpose(1, 0, 2, 3)
        )

        model_input = torch.from_numpy(
            model_input
        ).float().to(DEVICE)

        prediction = motionbert(
            model_input
        )

        # Test-time augmentation bằng flip.
        flipped_input = flip_h36m(
            model_input
        )

        flipped_prediction = motionbert(
            flipped_input
        )

        flipped_prediction = flip_h36m(
            flipped_prediction
        )

        prediction = (
            prediction
            + flipped_prediction
        ) / 2.0

        # Chuyển thành pose tương đối theo pelvis.
        prediction = (
            prediction
            - prediction[
                :, :, 0:1, :
            ]
        )

        # Người hoàn toàn không xuất hiện thì
        # đặt toàn bộ 3D pose về 0.
        person_exists = (
            h36m_scores.sum(
                axis=(0, 2)
            ) > 0
        )

        for person_index, exists in enumerate(
            person_exists
        ):
            if not exists:
                prediction[
                    person_index
                ] = 0.0

        # (M, T, J, 3)
        # -> (T, M, J, 3)
        joints_3d = (
            prediction
            .cpu()
            .numpy()
            .transpose(1, 0, 2, 3)
            .astype(np.float32)
        )

        return (
            joints_3d,
            h36m_scores.astype(
                np.float32
            ),
        )

    # %% Cell 11
    sample_joint_path = next(
        JOINTS_2D_ROOT.rglob(
            "*_joints.npy"
        )
    )

    relative_path = (
        sample_joint_path.relative_to(
            JOINTS_2D_ROOT
        )
    )

    sample_score_path = (
        SCORES_2D_ROOT
        / relative_path.parent
        / relative_path.name.replace(
            "_joints.npy",
            "_scores.npy",
        )
    )

    joints_2d = np.load(
        sample_joint_path
    ).astype(np.float32)

    scores_2d = np.load(
        sample_score_path
    ).astype(np.float32)

    joints_3d, scores_h36m = (
        lift_yolo_pose_to_3d(
            joints_2d,
            scores_2d,
        )
    )

    print("2D:", joints_2d.shape)
    print("3D:", joints_3d.shape)
    print("Scores:", scores_h36m.shape)
    print(
        "Finite:",
        np.isfinite(joints_3d).all(),
    )

    # %% Cell 12
    H36M_EDGES = [
        (0, 1),
        (1, 2),
        (2, 3),

        (0, 4),
        (4, 5),
        (5, 6),

        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),

        (8, 11),
        (11, 12),
        (12, 13),

        (8, 14),
        (14, 15),
        (15, 16),
    ]

    # %% Cell 13
    def show_pose_3d(
        joints_3d,
        scores,
        frame_index=None,
    ):
        if frame_index is None:
            frame_index = (
                len(joints_3d) // 2
            )

        figure = plt.figure(
            figsize=(12, 5)
        )

        for person_index in range(2):
            axis = figure.add_subplot(
                1,
                2,
                person_index + 1,
                projection="3d",
            )

            xyz = joints_3d[
                frame_index,
                person_index,
            ]

            confidence = scores[
                frame_index,
                person_index,
            ]

            for start, end in H36M_EDGES:
                if (
                    confidence[start] > 0
                    and confidence[end] > 0
                ):
                    axis.plot(
                        [
                            xyz[start, 0],
                            xyz[end, 0],
                        ],
                        [
                            xyz[start, 2],
                            xyz[end, 2],
                        ],
                        [
                            -xyz[start, 1],
                            -xyz[end, 1],
                        ],
                        linewidth=2,
                    )

            valid = confidence > 0

            axis.scatter(
                xyz[valid, 0],
                xyz[valid, 2],
                -xyz[valid, 1],
                c=confidence[valid],
                cmap="viridis",
            )

            axis.set_title(
                f"Person {person_index}"
            )

            axis.set_xlabel("X")
            axis.set_ylabel("Z")
            axis.set_zlabel("Y")

        plt.tight_layout()
        plt.show()

    # %% Cell 14
    show_pose_3d(
        joints_3d,
        scores_h36m,
    )

    # %% Cell 15
    joint_files = sorted(
        JOINTS_2D_ROOT.rglob(
            "*_joints.npy"
        )
    )

    print("Số clip:", len(joint_files))

    # %% Cell 16
    def atomic_save_npy(
        path,
        array,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        with open(
            temporary_path,
            "wb",
        ) as file:
            np.save(
                file,
                array,
            )

        os.replace(
            temporary_path,
            path,
        )

    # %% Cell 17
    success_count = 0
    error_records = []

    for joint_path in tqdm(
        joint_files,
        desc="MotionBERT 3D",
    ):
        relative_path = (
            joint_path.relative_to(
                JOINTS_2D_ROOT
            )
        )

        score_path = (
            SCORES_2D_ROOT
            / relative_path.parent
            / relative_path.name.replace(
                "_joints.npy",
                "_scores.npy",
            )
        )

        output_joint_path = (
            JOINTS_3D_ROOT
            / relative_path.parent
            / relative_path.name.replace(
                "_joints.npy",
                "_joints3d.npy",
            )
        )

        output_score_path = (
            SCORES_H36M_ROOT
            / relative_path.parent
            / relative_path.name.replace(
                "_joints.npy",
                "_scores_h36m.npy",
            )
        )

        # Resume nếu đã có output hợp lệ.
        if (
            output_joint_path.exists()
            and output_score_path.exists()
        ):
            try:
                existing_joints = np.load(
                    output_joint_path,
                    mmap_mode="r",
                )

                existing_scores = np.load(
                    output_score_path,
                    mmap_mode="r",
                )

                if (
                    existing_joints.ndim == 4
                    and existing_joints.shape[1:]
                    == (2, 17, 3)
                    and existing_scores.shape
                    == existing_joints.shape[:-1]
                ):
                    success_count += 1
                    continue

            except Exception:
                pass

        try:
            joints_2d = np.load(
                joint_path
            ).astype(np.float32)

            scores_2d = np.load(
                score_path
            ).astype(np.float32)

            joints_3d, h36m_scores = (
                lift_yolo_pose_to_3d(
                    joints_2d,
                    scores_2d,
                )
            )

            if not np.isfinite(
                joints_3d
            ).all():
                raise ValueError(
                    "Joints 3D chứa NaN/Inf"
                )

            atomic_save_npy(
                output_joint_path,
                joints_3d,
            )

            atomic_save_npy(
                output_score_path,
                h36m_scores,
            )

            success_count += 1

        except Exception as error:
            error_records.append(
                {
                    "joint_path": str(
                        joint_path
                    ),
                    "error": str(error),
                }
            )

    print("Thành công:", success_count)
    print("Lỗi:", len(error_records))

    for record in error_records[:10]:
        print(record)

    # %% Cell 18
    joint_3d_files = sorted(
        JOINTS_3D_ROOT.rglob(
            "*_joints3d.npy"
        )
    )

    invalid_files = []

    for path in tqdm(
        joint_3d_files,
        desc="Validate 3D",
    ):
        try:
            array = np.load(
                path,
                mmap_mode="r",
            )

            valid = (
                array.ndim == 4
                and array.shape[1:]
                == (2, 17, 3)
                and np.isfinite(array).all()
            )

            if not valid:
                invalid_files.append(
                    (
                        str(path),
                        array.shape,
                    )
                )

        except Exception as error:
            invalid_files.append(
                (
                    str(path),
                    str(error),
                )
            )

    print(
        "Số file 3D:",
        len(joint_3d_files),
    )

    print(
        "Không hợp lệ:",
        len(invalid_files),
    )
