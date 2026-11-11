# Pose-Based Badminton Stroke Recognition

Nhận dạng 18 động tác cầu lông từ chuỗi khớp người. Dự án có pipeline RTMPose/YOLO26 → pose 2D → MotionBERT pose 3D → ST-GCN, cùng các thí nghiệm ablation.

## Cấu trúc dự án

```text
badminton-stroke/
├── notebooks/                         # Khám phá, trực quan hóa và cấu hình thí nghiệm
│   ├── data-preprocessing.ipynb
│   ├── yolov26pose.ipynb
│   ├── motionbert_3d_extraction.ipynb
│   ├── stgcn_experiments.ipynb
│   ├── stgcn_experiments_motionbert.ipynb
│   └── stgcn_ablation.ipynb
├── src/badminton_stroke/              # Mã Python dùng lại
│   ├── paths.py                       # Đường dẫn dự án tập trung
│   ├── dataset.py                     # Dataset, augmentation và DataLoader
│   └── models/
│       ├── graphs.py                  # Đồ thị COCO và H36M
│       └── stgcn.py                   # SpatialGraphConv, STGCNBlock, STGCNV2
├── scripts/                           # Pipeline dữ liệu và trích xuất pose
│   ├── preprocess_data.py
│   ├── extract_yolo_pose.py
│   └── extract_motionbert_3d.py
├── experiments/                       # Chạy huấn luyện và đánh giá
│   ├── train_stgcn.py
│   ├── train_stgcn_motionbert.py
│   ├── run_stgcn_ablation.py
│   └── runs/                          # Kết quả mới, checkpoint và biểu đồ
├── models/                            # Trọng số đầu vào; yolo26s-pose.pt
├── data/                              # Video gốc (local)
├── processed/                         # Manifest, pose, confidence và kết quả cũ
├── third_party/MotionBERT/            # Mã, cấu hình và checkpoint MotionBERT
├── tools/export_notebooks.py          # Xuất và kiểm tra đồng bộ notebook → Python
├── tests/test_project_structure.py
├── docs/project_structure.md
├── dataset.py                         # Import tương thích cho mã cũ
├── pyproject.toml
└── uv.lock
```

Dữ liệu và kết quả cũ được giữ nguyên tại `processed/`; các lần huấn luyện mới ghi vào `experiments/runs/`. Việc tổ chức lại không xóa dữ liệu hoặc lịch sử Git.

## Môi trường

Python 3.11–3.12; dùng môi trường và phiên bản phụ thuộc trong `pyproject.toml` / `uv.lock`:

```bash
uv sync --frozen
source .venv/bin/activate
python -m jupyter lab
```

Mở notebook từ thư mục gốc hoặc `notebooks/`. Đường dẫn dữ liệu được tính từ vị trí source thay vì đường dẫn riêng của máy. Script có thể chạy bằng đường dẫn tuyệt đối từ thư mục khác.

## Chạy pipeline

Đặt dataset video theo lớp trong `data/`, với thư mục tên `VideoBadminton_Dataset`. Chạy lần lượt:

```bash
python scripts/preprocess_data.py --non-interactive
python scripts/extract_yolo_pose.py --non-interactive
python scripts/extract_motionbert_3d.py --non-interactive
python experiments/train_stgcn_motionbert.py --non-interactive
```

`preprocess_data.py` kế thừa cả phần tạo manifest và trích xuất RTMPose từ notebook gốc. Nếu chưa có `data.zip`, cần điền `GOOGLE_DRIVE_DATA_URL` trong notebook và xuất lại script. Với dữ liệu đã có, bước tải vẫn kiểm tra archive như notebook gốc; đọc hướng dẫn trong notebook trước khi chạy toàn bộ.

Trích xuất MotionBERT yêu cầu mã nguồn, cấu hình và checkpoint có sẵn trong `third_party/MotionBERT/`, cùng joints/confidence YOLO 2D trong `processed/`.

Các thí nghiệm khác:

```bash
python experiments/train_stgcn.py --non-interactive
python experiments/run_stgcn_ablation.py --non-interactive
```

Các script giữ thứ tự cell, bước thử mẫu, benchmark, biểu đồ và vòng lặp xử lý của notebook; chạy script sẽ chạy toàn bộ workflow. Tham số mô hình, số epoch và `EXPERIMENT_NAME` hiện vẫn được cấu hình trong notebook. `--non-interactive` chọn backend biểu đồ Agg; `--help` chỉ hiển thị hướng dẫn và không chạy pipeline.

## Cập nhật mã và kiểm tra

Mô hình và đường dẫn dùng chung được sửa trực tiếp trong `src/badminton_stroke/`. Các workflow được sửa trong notebook rồi xuất lại:

```bash
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m unittest discover -s tests -v
```

Để import package trong mã riêng mà chưa cài package, thêm `src/` vào `PYTHONPATH`:

```bash
PYTHONPATH=src python -c 'from badminton_stroke.models.stgcn import STGCNV2'
```

Chi tiết quy ước: [docs/project_structure.md](docs/project_structure.md).
