# Quy ước tổ chức mã và dữ liệu

## Phân chia trách nhiệm

- `notebooks/`: phân tích tương tác, cấu hình, thử nghiệm và biểu đồ. Giữ nguyên output cũ để tham khảo; output chỉ phản ánh lần chạy trước khi tái cấu trúc.
- `src/badminton_stroke/`: thư viện dùng lại, không tự chạy huấn luyện khi import. Mô hình ST-GCN dùng chung cho cả COCO 2D và H36M 3D, nhận adjacency qua constructor.
- `scripts/`: ba workflow xử lý dữ liệu xuất từ notebook.
- `experiments/`: ba workflow huấn luyện và đánh giá xuất từ notebook; `runs/` chứa kết quả sinh ra.
- `models/`: trọng số đầu vào, tách khỏi notebook. Checkpoint MotionBERT vẫn ở vị trí mà mã third-party yêu cầu.
- `tools/`: công cụ bảo trì và xuất notebook.
- `tests/`: kiểm tra cấu trúc, export, đường dẫn và forward mô hình.

Không sửa trực tiếp script được xuất nếu muốn giữ thay đổi qua lần export tiếp theo. Script có chú thích nguồn notebook và giữ ranh giới cell `# %%`. Các script chỉ bắt đầu workflow khi chạy trực tiếp, không khi import. Định nghĩa và logic phụ thuộc trạng thái cell vẫn nằm trong workflow; chỉ các thành phần đã tách vào `src/` được coi là API dùng lại.

## Đường dẫn và kết quả

`badminton_stroke.paths` là nguồn đường dẫn chung. Package hiện dùng theo source tree, chưa đóng gói wheel. Notebook tự thêm `src/` vào đường dẫn import; script tìm root từ `__file__`, sau đó chuyển working directory về root để tương thích mã notebook.

| Workflow | Đầu ra mới |
|---|---|
| Preprocessing / RTMPose | `processed/manifests/`, `processed/joints_npy_balanced/` và status CSV |
| YOLO26 | `processed/joints_npy_yolo26s_pose/`, `processed/scores_npy_yolo26s_pose/` và status CSV |
| MotionBERT | `processed/joints_3d_motionbert/`, `processed/scores_h36m_motionbert/` và cache |
| ST-GCN 2D | `experiments/runs/stgcn_v2/` |
| ST-GCN MotionBERT | `experiments/runs/stgcn_v2_yolo26_motionbert/` |
| Ablation | `experiments/runs/stgcn_ablation/<EXPERIMENT_NAME>/` |

Kết quả cũ trong `processed/stgcn*` không tự chuyển sang thư mục mới. Bảng so sánh ablation mới chỉ đọc `experiments/runs/stgcn_ablation/`. Các manifest và status cũ có thể chứa đường dẫn tuyệt đối theo máy; việc chuyển source không viết lại các tệp dữ liệu này.

`dataset.py` ở root giữ import tương thích; phần triển khai nằm trong `src/badminton_stroke/dataset.py`. Dataset này là implementation cũ; các dataset chuyên biệt của thí nghiệm vẫn nằm trong workflow vì cách tạo feature khác nhau.

## Tệp lớn và Git

Các tệp sinh ra trong `processed/`, `experiments/runs/` và trọng số trong `models/` được ignore cho các tệp mới. Những tệp đã được Git theo dõi trước đây vẫn được theo dõi; tái cấu trúc không thay đổi lịch sử hoặc untrack dataset hiện có.
