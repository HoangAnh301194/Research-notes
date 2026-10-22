# Báo cáo nghiên cứu ngày 17/10/2026

## A. Công việc đã làm
- Tổng hợp toàn bộ công việc đã làm , dữ liệu đã có để tiến hành viết sơ lược bài báo. 
   - Đánh giá **isolated và continuous gesture recognition** trên IPN-Hand.
   - Đo complexity, parameter của VideoMamba-Tiny và đối chiếu với VideoMAE / VideoMAEv2.

   - Đánh giá **isolated và continuous gesture recognition** trên CTPGesture.
   - Khảo sát các mức cấu hình **post-processing** theo thời gian: linking, loại đoạn ngắn, fill, smoothing.
   - So sánh với bài báo **CoboGesture** trên cùng IPN-Hand.
   - So sánh với một số bài SOTA của bộ dataset CTPGesture.

# 1. Cấu hình đã triển khai

## 1.1. VideoMamba-Tiny

Model trung tâm được sử dụng là **VideoMamba-Tiny**, với input video clip gồm 16 frame RGB.

| Thông số | Giá trị |
|---|---:|
| Backbone | VideoMamba-Tiny |
| Input | B × 3 × 16 × 224 × 224 |
| Số frame/model input | 16 |
| Precision | FP32 |
| Parameters | khoảng **6.96M** |
| Layers | 24 |
| Tokens/clip | 3,137 |
| Backend | native CUDA Mamba |
| GPU benchmark | NVIDIA Tesla T4 |
| PyTorch | 2.10.0 + CUDA 12.8 |

## 1.2. Complexity benchmark với VideoMAE

Các model được benchmark trên cùng input 1 × 3 × 16 × 224 × 224, FP32.

| Model | Params (M) | GFLOPs/clip | Mean latency (ms) | P95 latency (ms) | Throughput (clips/s) | Extra forward allocation |
|---|---:|---:|---:|---:|---:|---:|
| **VideoMamba-Tiny** | **6.9608** | **29.0261** | **48.48** | **49.45** | **20.63** | 56.58 MiB |
| VideoMAE v1 ViT-L/16 | 303.8751 | 596.8327 | 484.95 | 494.79 | 2.06 | 336.89 MiB |
| VideoMAE v2 ViT-L/16 | 303.8751 | 596.8327 | 459.3 | 461.9 | 2.18 | 336.89 MiB |

### Nhận xét tổng quan

- VideoMamba-Tiny nhẹ hơn khoảng **43.7 lần về số parameters**.
- Khối lượng tính toán ước lượng thấp hơn khoảng **20.6 lần**.
- Latency model-only thấp hơn khoảng **9.5–10 lần** so với VideoMAE ViT-L.
- Throughput **20.6 clips/s**, trong khi VideoMAE ViT-L khoảng 2 clips/s với cùng cấu hình hardware.

---

# 2. Thực nghiệm trên IPN-Hand

## 2.1. Dataset

IPN-Hand là benchmark continuous hand gesture recognition với:

- 200 video RGB;
- 50 subjects;
- hơn 800,000 frames;
- hơn 4,000 gesture instances;
- 13 gesture classes, cộng non-gesture;
- độ phân giải 640 × 480;
- 30 FPS.

## 2.2. Isolated gesture recognition trên IPN-Hand

Protocol đánh giá:

1. dùng ground-truth gesture interval;
2. lấy 16 frames rải đều trong mỗi đoạn;
3. một lần forward;
4. không sliding window;
5. không post-processing.

### Kết quả

| Metric | VideoMamba-Tiny |
|---|---:|
| Samples | 1,101 |
| **Top-1 Accuracy** | **84.65%** |
| Mean per-class accuracy | khoảng **77.26%** |
| Elapsed evaluation time | khoảng 252 s |
---

# 3. Đối chiếu IPN-Hand với bài CoboGesture

## 3.1. Bài báo đối chứng

**P.-D. Nguyen et al., “CoboGesture: A Continuous Hand Gesture Dataset and Recognition Method for Human-Collaborative Robot Interaction,” IEEE Access, vol. 14, pp. 27584–27601, 2026. DOI: 10.1109/ACCESS.2026.3664027.**
### Kết quả isolated của bài CoboGesture trên IPN-Hand

| Method trong CoboGesture | Accuracy | Params |
|---|---:|---:|
| ResNet-50 | 73.10% | 46.25M |
| C3D | 77.75% | 50.75M |
| RGB-CoGes + VideoMAE | 78.81% | 303.8M |
| RGB-CoGes + VideoMAEv2 | **84.59%** | 303.8M |
| **VideoMamba-Tiny (our method)** | **84.65%** | **6.96M** |

### So sánh

- VideoMamba-Tiny cao hơn VideoMAE: **+5.84 percentage points**.
- VideoMamba-Tiny cao hơn VideoMAEv2: **+0.06 percentage points**.
- VideoMamba chỉ có khoảng **1/43.7 số parameters** so với VideoMAE/VideoMAEv2 ViT-L.

---

# 4. Continuous recognition trên IPN-Hand

## 4.1. Sliding-window configuration

Cấu hình cuối đang dùng:

| Tham số | Giá trị |
|---|---:|
| FPS | 30 |
| Window size | **64 frames** |
| Step | **16 frames** |
| Overlap | **75%** |
| Model input | 16 frames |
| Sample offsets | 0, 4, 8, ..., 60 |
| Temporal context | khoảng 2.13 s |
| Prediction interval | khoảng 0.53 s |

## 4.2. Raw continuous result

| Metric | Raw |
|---|---:|
| Test videos | 52 |
| **Mean per-video Frame Accuracy** | **83.4889%** |
| **Global Frame Accuracy** | **83.7308%** |
| **Mean Temporal Overlap** | **49.1110%** |
| **Mean Gesture Recall** | **78.8995%** |

## 4.3. Post-processing ablation

| Config | T_AVG | MIN_GESTURE | Smooth | Mean FrameAcc | Global FrameAcc | Temporal Overlap | Gesture Recall |
|---|---:|---:|:---:|---:|---:|---:|---:|
| Raw | 0 | 0 | No | 83.4889% | 83.7308% | 49.1110% | **78.8995%** |
| L2 | 0.7 s | 0 | No | 83.5040% | 83.7478% | 49.2561% | 78.7206% |
| R2 | 0.7 s | 0.7 s | No | 83.5149% | 83.7572% | **50.7743%** | 76.3760% |
| **R3** | **0.7 s** | **1.0 s** | No | **83.5249%** | **83.7671%** | 50.7705% | 76.2844% |
| R4 | 0.7 s | 1.5 s | No | 83.3291% | ~83.57% | **51.1539%** | 72.4757% |
| S1 | 0.7 s | 1.0 s | Yes | 83.4537% | ~83.71% | 50.3951% | 75.0981% |

---

# 5. So sánh continuous IPN-Hand với CoboGesture

Theo Table 7 của bài CoboGesture:

| Method | Backbone | Frame-wise Accuracy | Activity Overlap Score |
|---|---|---:|---:|
| RGB-CoGes | VideoMAE | 80.00% | 0.370 |
| RGB-CoGes | VideoMAEv2 | 83.00% | **0.510** |
| **VideoMamba Raw** | VideoMamba-Tiny | **83.73%** | 0.4911 |
| **VideoMamba R3** | VideoMamba-Tiny | **83.77%** | **0.5077** |

### Nhận xét

So với RGB-CoGes + VideoMAEv2:

- Frame Accuracy của VideoMamba R3 cao hơn **+0.77 pp**.
- Activity overlap gần tương đương: **0.5077 so với 0.510**.
- Số parameters thấp hơn khoảng **43.7 lần**.
---

# 6. CTPGesture isolated evaluation

## 6.1. Isolated protocol

1. lấy ground-truth segment boundary;
2. sample đều 16 frames trên toàn segment;
3. không sliding window;
4. không post-processing.

| Metric | All classes 0–8 | Gesture-only 1–8 |
|---|---:|---:|
| Samples | 1,728 | 859 |
| **Accuracy** | **98.55%** | **98.02%** |
| Balanced Accuracy | 98.12% | 98.00% |
| Macro Precision | 97.87% | 98.54% |
| Macro Recall | 98.12% | 98.00% |
| **Macro-F1** | **97.96%** | **98.24%** |
| Weighted-F1 | 98.55% | 98.24% |

### Per-class F1

| Class | F1 |
|---|---:|
| No Gesture | 99.31% |
| Stop | 96.62% |
| Forward | 97.58% |
| **Left Turn** | **100.00%** |
| Left Turn Waiting | 99.12% |
| Right Turn | 99.07% |
| Lane Changing | 96.40% |
| Slow Down | 96.68% |
| Pull Over | 96.86% |

![CTPGesture isolated confusion matrix](../261015/cell17_output5.png)

# 7. CTPGesture continuous sliding-window evaluation

## 7.1. Cấu hình

Do video CTPGesture là 15 FPS, window được giảm từ 64 xuống 32 frame để vẫn giữ temporal context tương đương IPN-Hand.

| Tham số | IPN-Hand | CTPGesture |
|---|---:|---:|
| FPS | 30 | 15 |
| Window size | 64 | **32** |
| Step | 16 | **8** |
| Overlap | 75% | **75%** |
| Model frames | 16 | 16 |
| Sampling offsets | 0,4,...,60 | **0,2,...,30** |
| Context/window | ~2.13 s | **~2.13 s** |
| Prediction interval | ~0.53 s | **~0.53 s** |

## 7.2. CTPGesture continuous RAW result

| Metric | Giá trị |
|---|---:|
| Videos | 11 |
| Total frames | 67,356 |
| Total windows | 8,392 |
| **Global Frame Accuracy** | **83.98%** |
| Balanced Accuracy | 84.00% |
| Macro Precision | 78.52% |
| Macro Recall | 84.00% |
| **Macro-F1** | **80.61%** |
| Weighted-F1 | 83.96% |
| **Mean Gesture Recall** | **94.80%** |
| **Mean Temporal Overlap** | **65.32%** |
| Mean model forward | khoảng 48 ms/window |
| Effective evaluation throughput | khoảng 8.67 windows/s |

### Confusion matrix

![CTPGesture continuous RAW confusion matrix](../261015/cell23_output5.png)

## 7.3. Phân tích isolated → continuous

- Isolated Accuracy: **98.55%**.
- Isolated Macro-F1: **97.96%**.
- Continuous Macro-F1: **80.61%**.
- Continuous Gesture Recall: **94.80%**.
- Temporal Overlap: **65.32%**.

---

# 8. CTPGesture post-processing

| Config | T_AVG | MIN_GESTURE | Smooth | Global FrameAcc | Macro-F1 | Temporal Overlap | Gesture Recall |
|---|---:|---:|:---:|---:|---:|---:|---:|
| **Raw** | 0 | 0 | No | 83.98% | 80.61% | 65.32% | **94.80%** |
| **R3** | 0 | **1.0 s** | No | **83.99%** | **80.63%** | **65.90%** | 94.45% |
| S1 | 0 | 1.0 s | Yes | 83.81% | 79.71% | 64.85% | 91.57% |
| R4 | 0 | 1.5 s | No | 83.85% | 79.67% | 64.87% | 90.39% |

### Nhận xét

- R3 chỉ tăng Frame Accuracy khoảng **+0.01 pp**.
- Macro-F1 tăng khoảng **+0.01 pp**.
- Temporal Overlap tăng khoảng **+0.58 pp**.
- Gesture Recall giảm khoảng **−0.35 pp**.
---







# 9. Literature benchmark — CTPGesture v1

**Phạm vi:** CTPGesture **v1**; tách **isolated** và **continuous**; ưu tiên 2023–2026. **Parameters** tính theo toàn bộ mô hình; khi không thể tính đủ ghi rõ phần còn thiếu. `†`: tham chiếu kiến trúc hoặc suy ra từ nguồn khác, **không phải số đo của bài CTPGesture**. `---`: chưa xác minh.

## 9.1. Isolated — clip classification

| Method | Input | **Params (M)** | **GFLOPs** | Accuracy | Macro-F1 | Chứng cứ |
|---|---|---:|---:|---:|---:|---|
| **VideoMamba-Tiny (OURS — PROPOSED METHOD)** | **RGB, 16×224²** | **6,9598** | **29,026/clip** | **98,55%** | **97,96%** | **[Notebook cell 16](../261015/VideoMambaTinyCTPGestureFinal.ipynb)** |
| P3D (Qiu, benchmark 2021) | RGB | ≈65,3† (P3D-199) | --- | 92,21% | --- | [Xiong Table 3, PDF tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |
| C3D (Tran, benchmark 2021) | RGB | 78,4† | 38,5† (16×112²) | 87,12% | --- | [Xiong Table 3, PDF tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |
| 3D ResNet (Hara, benchmark 2021) | RGB | 33,3† (R3D-18) | --- | 81,16% | --- | [Xiong Table 3, PDF tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |

**Nguồn tham số:** [P3D Fig. 1](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf#page=1): 261 MB ⇒ ≈65,3M nếu weights FP32; [C3D MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md): 78,4M/38,5G ở 112×112; [3D ResNet-18 Table 2](https://www.mdpi.com/2076-3417/10/2/557): 33,3M. **Biến thể P3D/ResNet trong Xiong chưa đủ chi tiết để xác nhận tham số chính xác**; GFLOPs khác kích thước ảnh không so sánh trực tiếp. Xiong dùng **9 video test / 1.565 clips**, còn nhóm dùng **11 video test / 1.728 clips**. Đây là **RGB baseline lịch sử 2021**, không phải SOTA thuộc 2023–2026.

## 9.2. Continuous — sliding-window recognition

| Method | Input / protocol | **Params (M)** | **GFLOPs/window** | Accuracy | Macro-F1 | Chứng cứ |
|---|---|---:|---:|---:|---:|---|
| **VideoMamba-Tiny (OURS — PROPOSED METHOD)** | **RGB, window 32 / stride 8, sample 16×224²** | **6,9598** | **29,026** | **83,98%** | **80,61%** | **[Notebook cell 22](../261015/VideoMambaTinyCTPGestureFinal.ipynb)** |
| RSC-GestureNet (Li et al., 2026) | Pose + RGB descriptor, causal 32 / stride 4 | ≳2,12† *chưa gồm detector* | --- | 93,33 ± 0,24% | 91,71 ± 0,27% | [Table 1, PDF tr.9](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=9) |

**Tách thành phần RSC:** recognition network **815.270** tham số, tính từ [mã tác giả](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) với [cấu hình hidden=96, 17 joints, RGB=12](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/configs/train.yaml); BlazePose Lite **≈1,3M†** theo [tài liệu tổng hợp](https://www.mdpi.com/2076-3417/16/9/4497); tổng tạm tính **≈2,12M + pose detector**, **không phải total verified**. RSC dùng pose, và đánh giá cửa sổ causal/end-frame khác frame-level aggregation của nhóm: **không xếp hạng trực tiếp**.

## 9.3. Paper 2026 khác — chỉ để đối chiếu

| Method | Input | **Params toàn pipeline (M)** | GFLOPs | Accuracy | Chứng cứ |
|---|---|---:|---:|---:|---|
| Zhang et al., BiLSTM–Transformer (2026) | YOLO11m-Pose → keypoints → BiLSTM–Transformer | **≈20,99†** (=20,9 + 0,08641) | --- | 98,91% *(sequence-to-one, khác isolated)* | [Table 3, PDF tr.17](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=17) |

YOLO11m-**Pose** có **20,9M params, 71,5 GFLOPs/ảnh 640×640** theo [Ultralytics — Pose (COCO)](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md); head **86,41K params** từ bài Zhang. **71,5 GFLOPs/ảnh không phải GFLOPs/toàn sequence** nên để `---` trong cột tổng. Bài Zhang dùng 10 test videos và skeleton, không so sánh trực tiếp với 11-video RGB isolated.

**Tình trạng đối chứng:** Chưa xác minh được một nghiên cứu **RGB-only, 2023–2026, CTPGesture v1, cùng split và cùng protocol** có số liệu trực tiếp. [SlowFast 2024](https://doi.org/10.1117/12.3048304) chưa xác minh full text. Để kết luận hiệu quả của **Proposed Method**, cần tái huấn luyện các baseline RGB trên cùng split.

**PDF:** [RSC-GestureNet](./papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf) · [BiLSTM–Transformer](./papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf) · [Xiong 2021](../261015/electronics-10-00551-v2.pdf). [Confusion matrix isolated](../261015/cell17_output5.png) · [continuous](../261015/cell23_output5.png).

---

# 10. So sánh hai dataset đã chạy

| Hạng mục | IPN-Hand | CTPGesture v1 |
|---|---:|---:|
| Domain | hand micro-gesture | traffic-police full-body |
| FPS | 30 | 15 |
| Classes | 14 gồm non-gesture | 9 gồm no-gesture |
| Model | VideoMamba-Tiny | VideoMamba-Tiny |
| Input frames | 16 | 16 |
| Window | 64 | 32 |
| Step | 16 | 8 |
| Overlap | 75% | 75% |
| Temporal context | ~2.13 s | ~2.13 s |
| **Isolated Accuracy** | **84.65%** | **98.55%** |
| **Raw Global Frame Accuracy** | **83.73%** | **83.98%** |
| **Raw Temporal Overlap** | **49.11%** | **65.32%** |
| **Raw Gesture Recall** | **78.90%** | **94.80%** |


## B. Khó khăn
- Không

## C. Công việc tiếp theo
- Tiếp trục train với bộ EgoEv Gesture 
  - Đánh giá, so sánh với **CoboGesture HUST 2026**

- frame wise accuracy kiểm tra lại xem cách đánh giá acc có đúng không 
- tìm gflops từ các nguồn fill vào bảng 
- 4 bảng ( mỗi dataset 2 bảng ) , nếu thêm ego thì 6 bảng
- CTP gesture: 1 bài baseline video, 1 bài SOTA skeleton 
  - tìm khoảng 5 bài , kiến trúc , số liệu 
- IPN-Hand: tìm các bài gần đây , nonskeleton 
- EgoEv gesture : 
- liệt kê 1 bnarg mẫu trước , sau đó list từ các bài báo vào : param , acc, F1 , GFlops ,.... 
- accuracy@10 20 50%
- early@10%

- list ra các bài báo, gửi bảng tổng quan các bài cho THầy trước . 
- Deadline hoàn thành trong tối nay ( 22.10.2026)
