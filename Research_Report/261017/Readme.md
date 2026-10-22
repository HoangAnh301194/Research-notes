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







# 9. Khảo sát và đối chứng CTPGesture v1 (22/10/2023–22/10/2026)

**Phạm vi:** phương pháp dự đoán cử chỉ từ chuỗi **RGB video** trên đúng [CTPGesture v1 chính thức](https://github.com/zc402/traffic-gesture-datasets) (15 FPS, 1080×1080, nhãn từng frame, 9 lớp). Chỉ giữ tối đa **hai tài liệu pose/skeleton** để tham chiếu. Dấu **---** là *không báo cáo/không xác minh*, **không phải 0**. Số liệu của các bài dùng protocol khác **không được xem là xếp hạng SOTA trực tiếp**.

## 9.1. Kết quả hiện tại, đọc từ notebook ngày 15/10/2026

Nguồn: [VideoMambaTinyCTPGestureFinal.ipynb, cells 16/22/26](../261015/VideoMambaTinyCTPGestureFinal.ipynb) và [Readme ngày 15/10](../261015/Readme.md).

| Protocol | Đơn vị đánh giá | Test | Accuracy | Macro-F1 | Parameters | GFLOPs |
|---|---|---|---:|---:|---:|---:|
| **Isolated (VideoMamba-Tiny)** | 1 nhãn / đoạn GT | 1.728 đoạn, 11 video; 9 lớp | **98,55%** | **97,96%** | **6.959.817** | **29,026/clip** |
| Isolated, gesture-only | 1 nhãn / đoạn GT | 859 đoạn, 8 lớp, không tính lớp 0 | 98,02% | 98,24% | Như trên | Như trên |
| **Continuous RAW (VideoMamba-Tiny)** | frame-level sau tổng hợp cửa sổ | 67.356 frame, 11 video | **83,98%** | **80,61%** | **6.959.817** | **29,026/window** |
| Continuous R3 | frame-level, bỏ đoạn dự đoán <1 s | Cùng dữ liệu RAW | 83,99% | 80,63% | Như trên | Như trên |

Thông số tự đo: RGB 16×224×224, FP32, cửa sổ 32 frame / bước 8 (continuous), một lần forward trên 16 frame đã lấy mẫu, **Tesla T4**. GFLOPs = phép đếm *ước lượng trong notebook*, không đồng nhất mặc định với thống kê GFLOPs của các bài khác. Isolated dùng ranh giới GT; continuous không dùng ranh giới GT để tạo đầu vào. [Confusion matrix isolated](../261015/cell17_output5.png) · [Confusion matrix continuous](../261015/cell23_output5.png).

## 9.2. Hai nghiên cứu gần đây có nguồn dữ liệu liên quan được kiểm chứng

| Bài báo | Dữ liệu / điều kiện | Đầu vào recognition | Đơn vị báo cáo | Accuracy | Macro-F1 | **Toàn pipeline parameters** | **Toàn pipeline GFLOPs** | Chứng cứ trực tiếp |
|---|---|---|---|---:|---:|---|---|---|
| **Li et al. (2026), RSC-GestureNet** | **CTPGesture v1 chính thức**, 32-frame causal/stride 4; 11 test videos | **Pose + RGB descriptor** (không RGB-only) | Causal window/end-frame | **93,33 ± 0,24%** | **91,71 ± 0,27%** | --- | --- | [Table 1, PDF tr.9](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=9); [protocol, tr.5](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=5) |
| **Zhang et al. (2026), BiLSTM–Transformer** | *Police Gesture Dataset* liên quan V1, **10 test videos**, không trùng split chuẩn 11 video | **YOLOv11m-Pose → BiLSTM+Transformer** (không RGB-only) | Sequence-to-one (last-frame label), không phải toàn video | **98,91%** | --- (bài ghi “F1-score”, không xác định là Macro-F1) | **---**; chỉ head **86,41 K** được công bố | --- | [Table 1, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=12); [Table 3, tr.17](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=17) |
| **VideoMamba-Tiny (ours)** | **CTPGesture v1 chính thức**, 11 test videos | **RGB thuần** | **Isolated** | **98,55%** | **97,96%** | **6,959817 M** | **29,026/clip** | [Notebook cell 16](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| **VideoMamba-Tiny (ours)** | **CTPGesture v1 chính thức**, window 32 / stride 8 | **RGB thuần** | **Continuous frame-level** | **83,98%** | **80,61%** | **6,959817 M** | **29,026/window** | [Notebook cell 22](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |

**Không tính parameters thiếu:** 86,41 K trong [Zhang Table 3](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=17) là **recognition head**, **chưa tính YOLOv11m-Pose**. Bài RSC-GestureNet cũng không đưa ra số parameter/FLOPs toàn pipeline bao gồm MediaPipe. Không chuyển “model size” hoặc dung lượng checkpoint thành parameter. Bài Zhang ghi inference **1,3299 s/sequence nhưng không tính pose extractor**. Bài RSC công bố thêm **Online-F1 91,69 ± 0,29%** và TTC 0,153 ± 0,013 s ([Table 1](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=9)); TTC không phải độ trễ end-to-end.

**Giới hạn so sánh:** 98,91% của Zhang không phải isolated clip accuracy; 93,33% của RSC là causal-window accuracy chứ không phải frame accuracy được tổng hợp từ cửa sổ như VideoMamba. **Không lấy hiệu hai con số làm mức vượt trội.**

## 9.3. Nhóm RGB: kết quả có số liệu

Bài **Xiong et al. (2021), Electronics 10(5):551** đánh giá các mạng **RGB-video** trên *Police Gesture Dataset*, cắt clip từ nhãn GT, **11 train / 9 test videos**. Đây là **đối chứng lịch sử**, không phải bài SOTA thuộc giai đoạn 2023–2026 và cũng **không trùng split 11-video** của notebook.

| RGB baseline do Xiong et al. báo cáo | Clip Accuracy | Params đầy đủ (bài đối chứng) | GFLOPs | Bằng chứng |
|---|---:|---:|---:|---|
| **3D ResNet (Hara)** | 81,16% | --- | --- | [Table 3, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |
| **C3D (Tran)** | 87,12% | --- | --- | [Table 3, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |
| **P3D (Qiu)** | 92,21% | --- | --- | [Table 3, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |
| **ConvLSTM** | 82,40% | --- | --- | [Table 3, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12) |

**Không dùng MD-GCN 98,95% như RGB baseline:** phương pháp chính của Xiong sử dụng skeleton. Con số “**11,7 M**” của bài nằm trong cột **Model Size**, *không được xác định là 11,7 triệu parameters*, và chưa tính OpenPose: xem [Table 1, tr.10](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=10).

## 9.4. Nguồn đã sàng lọc nhưng không được đưa vào benchmark chính

| Công trình | Quyết định | Lý do | Nguồn |
|---|---|---|---|
| **Zhu & Fang (2024), SlowFast** | **Ứng viên RGB, chờ xác minh** | Có DOI/ấn phẩm SPIE nhưng chưa lấy được PDF toàn văn và chưa kiểm chứng được đúng V1/split/accuracy | [DOI 10.1117/12.3048304](https://doi.org/10.1117/12.3048304) |
| **Xiao et al. (2025), NTPGR** | **Không chọn (skeleton thứ ba)** | EPFFNet trích xuất pose; tác giả mô tả 21 video, test 10 video / 778 gesture instances khác protocol; Table 3 còn đối chiếu nhiều nghiên cứu vốn dùng sEMG/Wi-Fi, không xem là chứng cứ RGB trên V1 | [Table 2, tr.11](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/s41598-025-02833-y.pdf#page=11); [Table 3, tr.13](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/s41598-025-02833-y.pdf#page=13); [References, tr.18–19](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/s41598-025-02833-y.pdf#page=18) |
| **He et al. (2026), PR-Graph/EPS** | **Loại** | Báo cáo chính trên **CTPGesture v2**, không phải V1 | [DOI 10.1016/j.jvcir.2025.104698](https://doi.org/10.1016/j.jvcir.2025.104698) |
| **Liu et al. (2026), YOLOv8n lightweight** | **Loại khỏi clip/continuous** | Bài báo đánh giá **object detection mAP** trên dữ liệu cử chỉ, chưa kiểm chứng là đúng tập V1 và không phải cùng tác vụ classification/segmentation | [DOI 10.3785/j.issn.1008-973X.2026.08.007](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) |

**Kết quả sàng lọc:** chưa xác minh được đủ 3–5 bài RGB-only (2023–2026) *đồng thời* dùng **CTPGesture v1**, cung cấp **PDF toàn văn** và **số liệu đúng loại protocol**. Vì vậy **không lập bảng xếp hạng SOTA giả định**. Hướng bổ sung thực nghiệm đối chứng trực tiếp: fine-tune **C3D / 3D ResNet / P3D / SlowFast** trên **cùng 11-video test split**, cùng mẫu 16×224×224 (hoặc báo rõ khác biệt input), rồi đánh giá riêng isolated và continuous.

## 9.5. PDF có thể mở tới đúng bảng/figure

Hai PDF đã có từ báo cáo ngày 15/10 được **sao chép và đặt tên chuẩn**, giữ nguyên bản gốc (không thêm bản tải trùng):

- [Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf) — [Fig. 1, tr.6](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=6), [Table 1, tr.9](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=9), [Table 7, tr.13](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Li_Gao_Fu_2026_RSC-GestureNet_CTPGestureV1.pdf#page=13).
- [Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf) — [Fig. 5, tr.11](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=11), [Table 1, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=12), [Table 3, tr.17](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261017/papers/Zhang_et_al_2026_BiLSTM_Transformer_Pose.pdf#page=17).
- PDF lịch sử RGB: [Xiong et al. (2021), Table 3, tr.12](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/electronics-10-00551-v2.pdf#page=12).
- PDF NTPGR (đã có, không chọn benchmark chính): [Xiao et al. (2025), Table 2, tr.11](https://raw.githubusercontent.com/HoangAnh301194/Research-notes/main/Research_Report/261015/s41598-025-02833-y.pdf#page=11).

**Quy ước:** tất cả liên kết `#page=N` trỏ tới **trang PDF đếm từ 1**; hệ thống xem PDF của trình duyệt cần hỗ trợ fragment `#page`. Không sử dụng figure/table của bài khác để suy ra kết quả mới của nhóm.

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