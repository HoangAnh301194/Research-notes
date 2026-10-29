# Báo cáo nghiên cứu ngày 28/10/2026

## A. Công việc đã làm.

- Đóng gói dữ liệu **EgoEv Gesture** lên Kaggle, kiểm tra cấu trúc/split và huấn luyện mô hình **VideoMamba-Tiny**.
- Xuất keyframe cho các trường hợp dự đoán **đúng/sai** từ hai mô hình VideoMamba đã huấn luyện trên **IPN Hand** và **CTP Gesture**, đồng thời tổng hợp các chỉ số đánh giá.

### 1. Đóng gói dataset EgoEv Gesture

**Phương án chuẩn bị dữ liệu**

- Sử dụng bộ dữ liệu sự kiện **EgoEvGesture (38 lớp)** đã có trên Kaggle. Xây dựng bộ **Heterogeneous Cleaned Split v1** cố định theo subject để sử dụng thống nhất cho huấn luyện và đánh giá; **không sao chép lại dữ liệu sự kiện gốc** khi tạo split.
- **Kiểm tra trùng lặp:** đối chiếu Subject 5 và 10; với các file còn lại, sàng lọc theo kích thước, fingerprint và xác nhận trùng tuyệt đối bằng SHA-256. Kiểm tra nguy cơ file trùng xuất hiện giữa train/validation/test.

**Kết quả xử lý dữ liệu**

| Nội dung | Kết quả |
|---|---|
| Dữ liệu ban đầu | **5.398 mẫu**, 10 subjects |
| Subject trùng | **Subject 5 và 10:** 587 cặp có tên/kích thước khớp; loại **587 mẫu Subject 10** (SHA-256 của toàn bộ cặp được dẫn từ kiểm toán trước) |
| File trùng khác | **1 mẫu trùng nội bộ** |
| Dữ liệu sau làm sạch | **4.810 mẫu**, 9 subjects |
| **Train** | **2.626 mẫu**, subjects 3, 4, 5, 6, 7 |
| **Validation** | **489 mẫu**, subject 8 |
| **Test** | **1.695 mẫu**, subjects 1, 2, 9 |


**Cấu hình huấn luyện**

| Parameter | Giá trị theo notebook |
|---|---|
| Model | VideoMamba-Tiny, 38 lớp; khoảng **6,965 triệu tham số** |
| Input | Sự kiện Nx4 → 16 khung, 3 kênh, 224×224 |
| Channels | Số sự kiện dương, số sự kiện âm, tổng số sự kiện |
| Preprocessing | Chia 16 khoảng thời gian; log1p, chuẩn hóa percentile 99, ImageNet mean/std |
| Event limit | 200.000 sự kiện/mẫu |
| Hardware | 2 × Tesla T4, PyTorch DataParallel (không phải DDP) |
| Global batch / Gradient accumulation | 2 / 4 → effective batch **8** |
| Optimizer | AdamW; learning rate ban đầu **1e-4**, weight decay **0,05** |
| LR schedule | Warmup 3 epochs, cosine schedule; min LR **1e-6** |
| Drop path / Head dropout | 0,1 / 0,3 |
| Seed / Precision | 42 / FP32 (không AMP) |
| Training plan | 30 epochs; mỗi phiên tối đa 8 epochs, có resume checkpoint |

**Kết quả huấn luyện sau 16 epochs**

| Mốc | Train Accuracy | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|
| Epoch 9 — checkpoint tốt nhất | 98,97% | **69,53%** | **65,26%** |
| Epoch 14 | 99,35% | 67,69% | 64,67% |
| Epoch 16 — checkpoint mới nhất | **99,31%** | **58,90%** | **53,01%** |

- Đã hoàn thành **16/60 epochs**. Kết quả tốt nhất được ghi tại **epoch 9**
- Kết thúc phiên: lưu **checkpoint-best.pth (epoch 9)**, **checkpoint-latest.pth (epoch 16)**, cấu hình và lịch sử train/validation. Checkpoint chứa trạng thái optimizer để tiếp tục.
- **Nhận xét:** Train Accuracy gần 99% nhưng Validation Accuracy dao động và giảm còn 58,90% tại epoch 16 là **dấu hiệu quá khớp hoặc chênh lệch phân phối giữa subject train và validation**. Chưa đủ cơ sở xác định nguyên nhân duy nhất. Không được trình bày 69,53% là test Accuracy.


### 2. Xuất ra keyframe correct and error prediction để quan sát

Các hình keyframe được dùng để đối chiếu nhãn thực tế (GT) với nhãn dự đoán (Pred) và phân tích nguyên nhân nhận dạng/định vị thời gian chưa đúng. Kết quả isolated và continuous offline được báo cáo riêng do khác giao thức.

#### 2.1 IPN Hand

**Giao thức đánh giá**

- Dùng checkpoint đã fine-tune, **không train lại**. Mô hình nhận **16 RGB frames/clip, 224×224, 14 lớp**.
- **Isolated**: đánh giá **1.101 đoạn** từ tập test đã chú thích, loại nhãn D0X khỏi đánh giá isolated.
- **Continuous offline**: đánh giá **52 video, 212.574 frames**, cửa sổ **64 frames**, bước trượt **16 frames** (overlap 75%); lấy 16 khung làm đầu vào cho mỗi cửa sổ.
- Thử nghiệm early recognition bằng cách cắt một phần **đoạn đã biết biên ground truth** rồi lấy mẫu lại; **không phải đo thời gian phản hồi trực tuyến**.

**Kết quả định lượng**

| Chỉ số | Giá trị |
|---|---:|
| Isolated Top-1 Accuracy | **84,65%** |
| Isolated Macro-F1 | **78,34%** |
| Continuous RAW global Frame Accuracy | **83,73%** |
| Continuous RAW global Macro-F1 | **64,40%** |
| Continuous segment F1@10 / F1@25 / F1@50 | **79,45% / 77,58% / 69,91%** |
| Continuous mean edit score (MS-TCN convention) | **74,38%** |
| Mean temporal overlap (RAW, per-video) | **49,11%** |

**Độ chính xác nhận dạng khi chỉ quan sát một phần đoạn cử chỉ:**

| Tỷ lệ quan sát | 10% | 20% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Accuracy | 38,69% | 53,77% | 76,11% | 83,83% | 84,65% |

- Các lớp dễ nhầm hơn gồm **G02 (25,00%)**, **G01 (42,31%)** và **G08 (51,92%)**, dựa trên isolated per-class accuracy.
- Đã tạo hình gồm **đúng 16 khung được đưa vào mô hình**, kèm nhãn GT/Pred; isolated: **932 đúng, 169 sai**. Continuous: xuất **820 trường hợp ghép đúng, 706 trường hợp lỗi/không ghép được**, theo tiêu chí ghép nhãn và temporal IoU ≥ 0,5.

#### 2.2 CTP Gesture

**Giao thức đánh giá**

- Dùng checkpoint VideoMamba-Tiny **epoch 17**, **6.959.817 tham số**, **16 RGB frames/clip, 224×224**, **9 lớp**.
- Kiểm tra **11 video test chính thức**, **67.356 frames**, **1.728 GT segments** (trong đó 869 no-gesture và 859 gesture).
- Isolated: mỗi đoạn được gán nhãn là một clip. Continuous offline: cửa sổ **32 frames, stride 8**, overlap 75%, tổng **8.392 cửa sổ**; ghép xác suất dự đoán thành nhãn theo frame.
- Môi trường: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, Tesla T4, native Mamba CUDA; kiểm thử checkpoint/model forward đã **PASS**.

**Kết quả định lượng**

| Chỉ số | Giá trị |
|---|---:|
| Isolated Accuracy — 9 lớp | **98,55%** |
| Isolated Macro-F1 — 9 lớp | **97,96%** |
| Isolated gesture-only Accuracy — GT nhãn 1–8 | **98,02%** |
| Continuous RAW global Frame Accuracy | **83,98%** |
| Continuous RAW global Macro-F1 | **80,61%** |
| Continuous RAW mean per-video Frame Accuracy | **83,50%** |
| Continuous segment F1@10 / F1@25 / F1@50 | **96,05% / 94,75% / 85,55%** |
| Mean temporal overlap | **65,32%** |
| Mean gesture recall | **94,80%** |
| Mean matched onset / offset absolute error (tIoU ≥ 0,5) | **0,290 s / 0,446 s** |

**Early isolated Accuracy (9 lớp, dựa trên GT-prefix):**

| Tỷ lệ quan sát | 10% | 20% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Accuracy | 86,05% | 93,17% | 97,57% | 98,38% | 98,55% |

**Keyframe dự đoán đúng và sai**

- **Isolated:** xuất hình **1.728/1.728 clips**, gồm **1.703 đúng** và **25 sai**, không lỗi giải mã.
- **Continuous:** xuất **970/970 trường hợp**, gồm **725 đúng** và **245 lỗi**; mỗi hình chứa 16 khung đại diện từ cửa sổ mô hình cùng timeline GT/Pred. Đếm trường hợp continuous dựa trên ghép một-một cùng lớp với temporal IoU ≥ 0,5, **không phải 970 video độc lập**.
- Ví dụ isolated: [dự đoán đúng](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/isolated_16frame/correct/correct_00001.jpg) · [dự đoán sai](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/isolated_16frame/error/error_00002.jpg).
- Ví dụ continuous: [trường hợp đúng](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/continuous_16frame_timeline/correct/correct_00004.jpg) · [trường hợp lỗi](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/continuous_16frame_timeline/error/error_00001.jpg).
- Dữ liệu phục vụ tái kiểm chứng gồm classification report, isolated predictions, per-video CSV, early accuracy, phân tích biên đoạn và các bảng manifest từng trường hợp.


## B. Khó khăn
- Không
## C. Công việc tiếp theo
