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







# 9. Literature trên CTPGesture / Police Gesture Dataset

## 9.1. MD-GCN — Electronics 2021

**Xiong et al., “Traffic Police Gesture Recognition Based on Gesture Skeleton Extractor and Multichannel Dilated Graph Convolution Network”, Electronics, 2021.**

Pipeline:

    RGB video
      → pose estimation
      → Gesture Skeleton Extractor
      → MD-GCN
      → classification

Kết quả Table 3:

| Model | Accuracy |
|---|---:|
| 3D ResNet / Hara | 81.16% |
| C3D / Tran | 87.12% |
| P3D / Qiu | 92.21% |
| Zhang et al. | 91.18% |
| Pigou et al. | 91.04% |
| **MD-GCN** | **98.95%** |

MD-GCN cấu hình dilation 1,2,4 có:

- model size: **11.7M**;
- reported inference time: 3.0 s theo định nghĩa của bài.

> Protocol của P1 là clip-level trên split cũ 11 train / 9 test, không trùng split final của notebook hiện tại.

---

## 12.2. NTPGR — Scientific Reports 2025

**Xiao et al., “A Lightweight and Efficient Gesture Recognizer for Traffic Police Commands Using Spatiotemporal Feature Fusion”, Scientific Reports, 2025.**

- EPFFNet trích xuất pose/keypoint geometric feature.
- MSNet kết hợp CNN + LSTM + temporal/channel attention.
- Reported Accuracy: **97.56%** = 759/778 gesture instances.
- Average response time: **0.76 s** trên GTX TITAN XP.

Protocol là gesture-instance, không tương đương trực tiếp với frame-level continuous evaluation.

---

## 12.3. BiLSTM–Transformer — Electronics 2026

**Zhang et al., “A Traffic Police Gesture Recognition Method Based on BiLSTM-Transformer Architecture”, Electronics, 2026.**

Pipeline:

    RGB
      → YOLOv11m-Pose
      → 17 keypoints/frame
      → 40-frame sequence
      → BiLSTM + Transformer

Kết quả:

| Head | Accuracy |
|---|---:|
| BiLSTM | 96.48% |
| Transformer | 91.24% |
| **BiLSTM–Transformer** | **98.91%** |

- temporal head: 86.41K parameters;
- reported 1.3299 s/sequence;
- thời gian này không bao gồm toàn bộ cost pose extractor.

Bài chỉ liệt kê 10 test videos và thiếu 101 so với split 11-video đang dùng.

---

## 12.4. RSC-GestureNet — 2026 preprint

**Li et al., “RSC-GestureNet: Reliability-Aware Selective Causal Recognition of Chinese Traffic Police Gestures”, arXiv:2608.02200v1.**

Đây là đối chứng phù hợp nhất cho continuous CTPGesture:

- official CTPGesture v1 split;
- 9 classes;
- 32-frame causal window;
- stride 4;
- đánh giá 3 random seeds;
- không dùng future frame.

Kết quả:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| GRU | 88.82% | 87.10% |
| TCN | 89.88% | 88.29% |
| MD-GCN reimplementation | 90.08% | 88.48% |
| **RSC-GestureNet** | **93.33 ± 0.24%** | **91.71 ± 0.27%** |
| **VideoMamba RAW hiện tại** | **83.98%** | **80.61%** |

VideoMamba evaluator hiện tại chưa cùng protocol causal:

- VideoMamba: window 32 / stride 8, probability từ overlapping windows được phân xuống frame.
- RSC: causal window 32 / stride 4, prediction end-aligned.

Vì vậy chưa được kết luận trực tiếp “RSC hơn VideoMamba 9.35%” theo nghĩa fair benchmark.

---

# 13. Bảng literature context cho CTPGesture isolated

| Method | Year | Representation | Params / size được report | Accuracy | Protocol |
|---|---:|---|---:|---:|---|
| 3D ResNet | 2021 | RGB | NR | 81.16% | legacy 9-video test |
| C3D | 2021 | RGB | NR | 87.12% | legacy 9-video test |
| P3D | 2021 | RGB | NR | 92.21% | legacy 9-video test |
| NTPGR | 2025 | pose + temporal | NR | 97.56% | 778 gesture instances |
| **VideoMamba-Tiny** | 2026 | **RGB** | **6.96M** | **98.55%** | **11 test videos / 9 classes** |
| BiLSTM–Transformer | 2026 | pose + temporal | head 86.41K | 98.91% | 10 test videos |
| MD-GCN | 2021 | skeleton GCN | 11.7M | 98.95% | legacy 9-video test |

---

# 14. So sánh hai dataset đã chạy

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