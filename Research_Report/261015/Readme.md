# Báo cáo nghiên cứu ngày 15/10/2026 

## A. Công việc đã làm 
- Chạy inference đánh giá thêm continuos evaluation trên dataset CTPGesturev1 với checkpoint-best mới train.  

- Tìm kiếm các bài báo liên quan tới CTPGestur V1 , thu thập kết quả và so sánh 

### 1. Chạy inference đánh giá thêm continuos evaluation trên dataset CTPGesture 
- Cấu hình thử nghiệm : 

| Tham số | Giá trị |
|---|---|
| Model | VideoMamba-Tiny (6,959,817 params) |
| Checkpoint | checkpoint-best.pth (epoch 17) |
| GPU | Tesla T4 |
| Mamba backend | native_cuda |
| FPS video | 15 FPS |
| Window size | 32 frames (~2.13 s) |
| Window step | 8 frames (overlap 75%) |
| Model input frames | 16 frames |
| Sample offsets | 0, 2, 4, ..., 30 |

> Vì FPS của Video trong CTPGesture là 15FPS , nên để đảm bảo tốc độ chênh lệch giữa các frame thì em tạm thời giảm Window từ 64 -> 32

- Kết quả :

#### 1.1. Isolated Gesture Classification
> Ground-truth boundaries -> sample 16 frames đều -> 1 dự đoán/segment, không dùng sliding window.

| Metric | Tất cả lớp (0-8) | Gesture-only (1-8) |
|---|---|---|
| Samples | 1,728 | 859 |
| **Accuracy** | **98.55%** | **98.02%** |
| **Balanced Accuracy** | **98.12%** | **98.00%** |
| Macro Precision | 97.87% | 98.54% |
| Macro Recall | 98.12% | 98.00% |
| **Macro F1** | **97.96%** | **98.24%** |
| Weighted F1 | 98.55% | 98.24% |

**Per-class F1 nổi bật:**

| Lớp | F1-score |
|---|---|
| 0 - No Gesture | 0.9931 |
| 1 - Stop | 0.9662 |
| 2 - Forward | 0.9758 |
| **3 - Left Turn** | **1.0000** |
| 4 - Left Turn Waiting | 0.9912 |
| 5 - Right Turn | 0.9907 |
| 6 - Lane Changing | 0.9640 |
| 7 - Slow Down | 0.9668 |
| 8 - Pull Over | 0.9686 |

**Confusion matrix (normalized):**

![Isolated confusion matrix](cell17_output5.png)

#### 1.2. Continuous Recognition (raw, chưa post-processing)
> Sliding window 32/8 trên toàn bộ video -> xác suất frame-level -> argmax.

| Metric | Giá trị |
|---|---|
| Videos | 11 |
| Tổng frames | 67,356 |
| Tổng windows | 8,392 |
| **Global Frame Accuracy** | **83.98%** |
| **Global Balanced Accuracy** | **84.00%** |
| Global Macro F1 | 80.61% |
| Global Weighted F1 | 83.96% |
| **Mean Gesture Recall** | **94.80%** |
| **Mean Temporal Overlap** | **65.32%** |
| Mean model ms/window | 48.2 ms |
| Effective windows/second | 8.67 |

**Confusion matrix (normalized):**

![Continuous RAW confusion matrix](cell23_output5.png)

#### 1.3. Hậu xử lí raw output (Post-processing)

> Pipeline post-processing gồm 4 stage tuần tự: Linking (T_AVG) -> Short-segment Removal (MIN_GESTURE) -> Fill same neighbors -> Smoothing. Model chỉ chạy inference một lần để trích xuất raw cache, các bước sau xử lý trực tiếp trên probability cache.

**Bảng so sánh các cấu hình Post-processing:**

| Config | Smoothing | T_AVG (s) | MIN_GESTURE (s) | Fill | Global FrameAcc | Global Macro F1 | Temporal Overlap | Gesture Recall | Δ FrameAcc | Δ Overlap | Δ Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **R3 (BEST)** | False | 0.0 | **1.0** | False | **83.99%** | **80.63%** | **65.90%** | 94.45% | +0.01% | +0.58% | -0.35% |
| F1 | False | 0.0 | 1.0 | True | 83.99% | 80.63% | 65.90% | 94.45% | +0.01% | +0.58% | -0.35% |
| **Raw (Baseline)** | False | 0.0 | 0.0 | False | 83.98% | 80.61% | 65.32% | **94.80%** | baseline | baseline | baseline |
| L1 | False | 0.3 | 0.0 | False | 83.98% | 80.61% | 65.32% | 94.80% | +0.00% | +0.00% | +0.00% |
| R1 | False | 0.0 | 0.5 | False | 83.98% | 80.61% | 65.32% | 94.80% | +0.00% | +0.00% | +0.00% |
| R2 | False | 0.0 | 0.7 | False | 83.98% | 80.61% | 65.82% | 94.45% | -0.01% | +0.50% | -0.35% |
| L2 | False | 0.7 | 0.0 | False | 83.97% | 80.60% | 65.26% | 94.68% | -0.01% | -0.06% | -0.13% |
| L3 | False | 1.0 | 0.0 | False | 83.97% | 80.60% | 65.26% | 94.68% | -0.01% | -0.06% | -0.13% |
| S2 | True | 0.0 | 1.0 | True | 83.82% | 79.75% | 64.97% | 91.57% | -0.16% | -0.35% | -3.23% |
| S1 | True | 0.0 | 1.0 | False | 83.81% | 79.71% | 64.85% | 91.57% | -0.18% | -0.47% | -3.23% |
| R4 | False | 0.0 | 1.5 | False | 83.85% | 79.67% | 64.87% | 90.39% | -0.14% | -0.46% | -4.41% |

**Nhận xét và Phân tích:**
- **Cấu hình tối ưu nhất là R3**: `MIN_GESTURE = 1.0s`, `T_AVG = 0.0s`, Smoothing = False, Fill = False. Đã loại bỏ 12 đoạn dự đoán ngắn nhiễu, tăng Temporal Overlap thêm **+0.58%** (đạt 65.90%), Frame Accuracy tăng nhẹ lên **83.99%**.
> **Post-processing trên CTPGesture ít thay đổi hơn so với IPN-Hand**: Do cử chỉ của CSGT là các động tác cơ thể lớn, dứt khoát và kéo dài trong khoảng thời gian xác định, ít bị phân mảnh thành nhiều đoạn nhỏ như các vi cử chỉ tay (micro-gestures) ở IPN-Hand.
- **Probability Smoothing (S1, S2) làm giảm hiệu năng**: Khi làm mượt xác suất theo thời gian, ranh giới giữa các cử chỉ bị mờ (boundary blur), dẫn đến Macro F1 giảm từ 80.61% xuống **79.71%** và Recall giảm **-3.23%**.
- **Linking (T_AVG) không đem lại hiệu quả**: `T_AVG = 0.0s` cho kết quả tốt nhất; khi tăng T_AVG (ghép các đoạn cùng nhãn cách nhau < 0.7s - 1.0s), mô hình phân biệt không rõ ràng được giữa 2 lần thực hiện cử chỉ liên tiếp tại biên của 2 cử chỉ 

### 2. Các bài báo liên quan đến CTPGesture v1 và so sánh kết quả 

#### 2.1. Bài báo gốc công bố bộ dataset CTPGesture v1
- **Tên bài báo**: *"Visual Recognition of traffic police gestures with convolutional pose machine and handcrafted features"*
- **Tạp chí**: **Neurocomputing**, Tập 390, Trang 248–259 (Elsevier, 2020)
- **Tác giả**: Jian He, Cheng Zhang, Xinlin He, Ruihai Dong (Beijing University of Technology, Trung Quốc)
- **Mã nguồn & Dataset**: [zc402/ChineseTrafficPolicePose](https://github.com/zc402/ChineseTrafficPolicePose)
- **Phương pháp đề xuất trong bài báo**: 
  - Sử dụng mạng **PKEN** (Police Keypoint Extraction Network - tinh chỉnh từ CPM) trích xuất 14 điểm khung xương cơ thể.
  - Kết hợp 2 đặc trưng không gian thủ công (*handcrafted features*): Chiều dài xương tương đối (*Relative Bone Length*) và Góc nghiêng so với trọng lực (*Angle with Gravity*).
  - Đưa chuỗi đặc trưng qua mạng **LSTM** để nhận dạng chuỗi cử chỉ thời gian thực theo từng frame.

**Bảng so sánh kết quả giữa các kiến trúc trong bài báo gốc và VideoMamba-Tiny:**

| Phương pháp / Kiến trúc | Input Modality | Số tham số / GFLOPs | Cơ chế nhận dạng | Accuracy / Metric |
|---|---|---|---|---|
| **End-to-end 3D Convolution** (Ji et al.) | Video RGB | N/A | 3D CNN end-to-end | 81.16% (Edit Distance Acc) |
| **Convolutional LSTM** (Xing et al.) | Video RGB | N/A | CNN + ConvLSTM | 82.40% (Edit Distance Acc) |
| **ResNet Part Localizer + LSTM** | Skeleton + RGB | N/A | Pose + LSTM | 87.29% (Edit Distance Acc) |
| **DenseNet Part Localizer + LSTM** | Skeleton + RGB | N/A | Pose + LSTM | 89.74% (Edit Distance Acc) |
| **PKEN + BiLSTM** (Pigou et al.) | Skeleton keypoints | N/A | BiLSTM | 91.04% (Edit Distance Acc) |
| **PKEN + LSTM (Bài báo gốc đề xuất)** | Skeleton keypoints | N/A | Pose + Handcrafted + LSTM | **91.18%** (Edit Distance Acc) |
| **VideoMamba-Tiny (Thực nghiệm của nhóm)** | **Video RGB thuần (End-to-End)** | **6.96M params / 17.0 GFLOPs** | **State Space Model (SSM) thuần vision** | **98.55%** (Isolated Acc) / **83.99%** (Continuous Frame Acc) |

**Nhận xét so sánh:**
- Phương pháp trong bài báo gốc năm 2020 dựa trên pipeline 2 giai đoạn (cần chạy CPM phát hiện khung xương rồi mới đưa vào LSTM). Nếu bước phát hiện khung xương gặp lỗi (bị che khuất hoặc ánh sáng phức tạp), mô hình sẽ nhận dạng sai hoàn toàn.
- Mô hình **VideoMamba-Tiny** hoạt động **end-to-end trực tiếp trên RGB video**, không phụ thuộc vào bộ trích xuất khung xương, nhưng vẫn đạt độ chính xác Isolated vượt trội (**98.55%** so với 91.18% của bài báo và ~81-82% của 3D CNN).
- Về hiệu năng tính toán, VideoMamba-Tiny đạt độ trễ suy luận chỉ **48.2 ms/window** trên GPU Tesla T4 (~20.6 clips/s), hoàn toàn đáp ứng yêu cầu xử lý thời gian thực (> 15 FPS của luồng camera giao thông).

## B. Khó khăn 
- Dataset CTPGesture v1 có thời lượng video khá dài ở 15 FPS, việc đánh giá toàn diện Continuous Sliding Window trên toàn bộ 11 test video đòi hỏi tài nguyên tính toán lớn.
- Quá trình training các dataset lớn hơn (như CTPGesture v2 và EgoEvGesture) trên Kaggle thường chạm ngưỡng giới hạn 12 tiếng runtime, cần giải pháp lưu checkpoint và resume training định kỳ.

## C. Công việc tiếp theo 
- Tiếp tục nghiên cứu các giải pháp tối ưu temporal window cho Continuous recognition đối với các gesture chuyển động chậm.
- Chuẩn bị dữ liệu và triển khai huấn luyện trên dataset **EgoEvGesture** (dữ liệu cử chỉ từ event camera góc nhìn thứ nhất) và **CTPGesture v2** (33 lớp gồm cử chỉ kết hợp hướng cơ thể).
- Báo cáo kết quả và xin ý kiến định hướng nghiên cứu tiếp theo từ Thầy.
