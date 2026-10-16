# Báo cáo nghiên cứu ngày 17/10/2026

## A. Tổng quan công việc đã thực hiện

Trong giai đoạn vừa qua, em tập trung triển khai và đánh giá pipeline nhận dạng cử chỉ bằng **VideoMamba-Tiny** trên hai bộ dữ liệu chính là **IPN-Hand** và **CTPGesture v1**, đồng thời bắt đầu chuẩn bị dữ liệu **EgoEvGesture** cho giai đoạn tiếp theo.

Các nhóm công việc chính đã hoàn thành:

- Đánh giá **isolated gesture recognition** trên IPN-Hand.
- Đánh giá **continuous gesture recognition** bằng sliding window trên IPN-Hand.
- Khảo sát post-processing theo thời gian: linking, loại đoạn ngắn, fill, smoothing.
- So sánh với bài báo **CoboGesture** trên cùng IPN-Hand.
- Đo complexity của VideoMamba-Tiny và đối chiếu với VideoMAE / VideoMAEv2.
- Fine-tune VideoMamba-Tiny trên **CTPGesture v1**.
- Đánh giá CTPGesture theo cả isolated và continuous protocol.
- Khảo sát post-processing trên CTPGesture.
- Khảo sát các công trình đã dùng Police Gesture Dataset / CTPGesture v1.
- Chuẩn bị và audit dữ liệu **EgoEvGesture** để phục vụ thí nghiệm tiếp theo.

> **Lưu ý về cách đọc kết quả:** các bảng literature bên dưới được tách theo từng protocol. Không gộp clip-level accuracy, gesture-instance accuracy và frame-level causal accuracy thành một leaderboard SOTA duy nhất vì các bài dùng split, modality và đơn vị đánh giá khác nhau.

---

# 1. Backbone và cấu hình triển khai chung

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

### Nhận xét

- VideoMamba-Tiny nhẹ hơn khoảng **43.7 lần về số parameters**.
- Khối lượng tính toán ước lượng thấp hơn khoảng **20.6 lần**.
- Latency model-only thấp hơn khoảng **9.5–10 lần** so với VideoMAE ViT-L.
- Throughput khoảng **20.6 clips/s**, trong khi VideoMAE ViT-L khoảng 2 clips/s.

> VideoMAE v1/v2 trong benchmark complexity được dựng để đo **kiến trúc và runtime**, không dùng checkpoint nhận dạng để so Accuracy. Vì vậy bảng trên chỉ dùng cho complexity, không dùng để suy ra chất lượng nhận dạng.

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

Trong notebook đánh giá hiện tại:

- **Isolated:** 1,101 annotated gesture clips.
- **Continuous:** 52 test videos.

Nguồn dataset: https://gibranbenitez.github.io/IPN_Hand/

---

## 2.2. Isolated gesture recognition trên IPN-Hand

Protocol:

1. dùng ground-truth gesture interval;
2. lấy đúng 16 frames trong mỗi đoạn;
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

Một số gesture nhỏ như click/point bằng 1–2 ngón khó hơn rõ rệt so với các gesture chuyển động lớn.

---

# 3. Đối chiếu IPN-Hand với bài CoboGesture

## 3.1. Bài báo đối chứng

**P.-D. Nguyen et al., “CoboGesture: A Continuous Hand Gesture Dataset and Recognition Method for Human-Collaborative Robot Interaction,” IEEE Access, vol. 14, pp. 27584–27601, 2026. DOI: 10.1109/ACCESS.2026.3664027.**

Bài báo sử dụng IPN-Hand làm benchmark ngoài để đánh giá framework RGB-CoGes, do đó đây là đối chứng phù hợp với pipeline hiện tại.

### Kết quả isolated của bài CoboGesture trên IPN-Hand

| Method trong CoboGesture | Accuracy | Params |
|---|---:|---:|
| ResNet-50 | 73.10% | 46.25M |
| C3D | 77.75% | 50.75M |
| RGB-CoGes + VideoMAE | 78.81% | 303.8M |
| RGB-CoGes + VideoMAEv2 | **84.59%** | 303.8M |
| **VideoMamba-Tiny của em** | **84.65%** | **6.96M** |

### Chênh lệch

- VideoMamba-Tiny cao hơn VideoMAE: **+5.84 percentage points**.
- VideoMamba-Tiny cao hơn VideoMAEv2: **+0.06 percentage points**.
- VideoMamba chỉ có khoảng **1/43.7 số parameters** so với VideoMAE/VideoMAEv2 ViT-L.

Đây là kết quả đáng chú ý: model rất nhỏ nhưng isolated accuracy vẫn tương đương VideoMAEv2.

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

Pipeline:

    Continuous video
          ↓
    Sliding window: 64 frames, step 16
          ↓
    Sample 16 frames
          ↓
    Resize / Crop / Normalize
          ↓
    VideoMamba-Tiny
          ↓
    Softmax probability/window
          ↓
    Aggregate probability xuống từng frame
          ↓
    Argmax frame-level labels
          ↓
    Optional temporal post-processing
          ↓
    Final segments

Ảnh khảo sát post-processing trước đó:

![IPN post-processing overview](../261007/image.png)

![IPN post-processing example](../261007/image-1.png)

## 4.2. Raw continuous result

| Metric | Raw |
|---|---:|
| Test videos | 52 |
| **Mean per-video Frame Accuracy** | **83.4889%** |
| **Global Frame Accuracy** | **83.7308%** |
| **Mean Temporal Overlap** | **49.1110%** |
| **Mean Gesture Recall** | **78.8995%** |

## 4.3. Post-processing ablation

Pipeline hậu xử lí:

    Linking (T_AVG)
      → Short-segment Removal (MIN_GESTURE)
      → Fill same neighbors
      → Smoothing

| Config | T_AVG | MIN_GESTURE | Smooth | Mean FrameAcc | Global FrameAcc | Temporal Overlap | Gesture Recall |
|---|---:|---:|:---:|---:|---:|---:|---:|
| Raw | 0 | 0 | No | 83.4889% | 83.7308% | 49.1110% | **78.8995%** |
| L2 | 0.7 s | 0 | No | 83.5040% | 83.7478% | 49.2561% | 78.7206% |
| R2 | 0.7 s | 0.7 s | No | 83.5149% | 83.7572% | **50.7743%** | 76.3760% |
| **R3** | **0.7 s** | **1.0 s** | No | **83.5249%** | **83.7671%** | 50.7705% | 76.2844% |
| R4 | 0.7 s | 1.5 s | No | 83.3291% | ~83.57% | **51.1539%** | 72.4757% |
| S1 | 0.7 s | 1.0 s | Yes | 83.4537% | ~83.71% | 50.3951% | 75.0981% |

### Nhận xét

- R3 được chọn nếu ưu tiên Frame Accuracy.
- Temporal Overlap tăng từ **49.11% → 50.77%**.
- Đổi lại Gesture Recall giảm từ **78.90% → 76.28%**.
- MIN_GESTURE quá lớn làm mất gesture thật.
- Smoothing không cho cải thiện đáng kể.

> Vì các cấu hình hậu xử lí được khảo sát trực tiếp trên test output, phần này nên trình bày là **ablation/thăm dò**, không dùng như tuned final protocol cho publication.

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

- Frame Accuracy của VideoMamba R3 cao hơn khoảng **+0.77 pp**.
- Activity overlap gần tương đương: **0.5077 so với 0.510**.
- Số parameters thấp hơn khoảng **43.7 lần**.

> Đây là **contextual comparison**, chưa phải reproduction tuyệt đối vì implementation sliding window/post-processing không hoàn toàn giống code của bài CoboGesture.

---

# 6. CTPGesture v1

## 6.1. Dataset

CTPGesture v1 gồm chuỗi video cảnh sát giao thông thực hiện liên tục 8 nhóm hiệu lệnh.

| Thông tin | Giá trị |
|---|---:|
| Tổng video | 21 |
| FPS | 15 |
| Classes | 9 |
| Annotated segments | 3,354 |
| Annotated frames | 134,424 |

Các class:

| ID | Class |
|---:|---|
| 0 | No Gesture / Stand in attention |
| 1 | Stop |
| 2 | Forward |
| 3 | Left Turn |
| 4 | Left Turn Waiting |
| 5 | Right Turn |
| 6 | Lane Changing |
| 7 | Slow Down |
| 8 | Pull Over |

## 6.2. Split sử dụng

| Split | Video IDs | Segments | Annotated Frames |
|---|---|---:|---:|
| Train | 001, 005, 007, 009, 013, 015, 017, 103 | 1,284 | 51,824 |
| Val | 003, 011 | 342 | 15,244 |
| Test | 002, 004, 008, 010, 012, 014, 016, 018, 101, 102, 104 | 1,728 | 67,356 |
| **Tổng** | 21 videos | **3,354** | **134,424** |

Provider train được tách thêm validation để tránh dùng official test trong quá trình training.

---

# 7. Fine-tune VideoMamba-Tiny trên CTPGesture

## 7.1. Training configuration

| Hyperparameter | Giá trị |
|---|---|
| Model | VideoMamba-Tiny |
| Pretrained | videomamba_t16_in1k.pth |
| Input | 16 frames × 224 × 224 |
| Sampling rate | 2 |
| Patch size | 16 × 16 |
| Tubelet size | 1 |
| Optimizer | AdamW |
| Beta | 0.9 / 0.999 |
| LR | 1e-4 |
| Minimum LR | 1e-6 |
| LR schedule | cosine decay |
| Warmup | 5 epochs |
| Weight decay | 0.05 |
| Layer decay | 0.75 |
| Planned epochs | 30 |
| Completed | 28 epochs trước Kaggle timeout |
| Gradient clipping | 1.0 |
| Effective batch size | 8 |
| Precision | FP32 |
| RandAugment | m5-n2 |
| ColorJitter | 0.4 |
| Random Erasing | p = 0.25 |
| Mixup / CutMix | OFF / OFF |
| Label smoothing | 0 |
| FC dropout | 0.3 |
| DropPath | 0.1 |
| Horizontal flip | OFF |
| Hardware | 2 × Tesla T4 |
| Distributed training | DDP + NCCL |

Horizontal flip được tắt vì Left Turn / Right Turn là các class khác nhau.

## 7.2. Training result

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Ghi chú |
|---:|---:|---:|---:|---:|---|
| 1 | 2.0191 | 33.8% | 1.7178 | 50.3% | warmup |
| 5 | 0.5631 | 84.6% | 0.5259 | 83.9% | cuối warmup |
| 7 | 0.4675 | 86.9% | 0.4257 | 87.4% | first best |
| **18** | **0.3610** | **91.2%** | **0.2325** | **92.1%** | **best checkpoint** |
| 26 | 0.2335 | **94.0%** | 0.3970 | 90.9% | train acc cao nhất |
| 28 | 0.2540 | 93.4% | 0.3838 | 90.6% | cuối trước timeout |

Best checkpoint nằm ở **epoch thứ 18** nếu đếm từ 1; trong checkpoint/code có thể lưu index epoch = 17 do đếm từ 0.

---

# 8. CTPGesture isolated evaluation

## 8.1. Giai đoạn 12-view đã thử

Trước continuous evaluation, em từng khảo sát multi-view evaluation:

| Sampling | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Single View | 97.92% | 97.80% | 97.45% | 97.93% |
| **12-View** | **98.50%** | **98.38%** | **98.01%** | **98.50%** |

12-view = 4 temporal positions × 3 spatial crops.

| Single View | 12-View |
|---|---|
| ![Single view confusion matrix](../261007/single_view_cm.png) | ![12-view confusion matrix](../261007/12view_cm.png) |

## 8.2. Final isolated protocol

Protocol final dùng để báo cáo chính:

1. lấy ground-truth segment boundary;
2. sample đều 16 frames trên toàn segment;
3. một lần inference;
4. không sliding window;
5. không 12-view;
6. không post-processing.

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

### Confusion matrix

![CTPGesture isolated confusion matrix](../261015/cell17_output5.png)

### Nhận xét

- Khả năng phân biệt class khi đã biết đúng temporal boundary là rất cao.
- Các class khó hơn gồm Lane Changing, Pull Over, Stop, Slow Down.
- Kết quả này cho thấy classifier/backbone không phải bottleneck chính của bài toán continuous.

---

# 9. CTPGesture continuous sliding-window evaluation

## 9.1. Cấu hình

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

Pipeline continuous:

    Full video
       ↓
    Sequential decode
       ↓
    Window = 32, step = 8
       ↓
    Sample 16 frames/window
       ↓
    Batch inference
       ↓
    Probability/window
       ↓
    Average probability từ mọi window phủ lên frame
       ↓
    Frame-level argmax
       ↓
    Optional post-processing
       ↓
    Temporal metrics

## 9.2. Tối ưu tốc độ evaluator

Implementation đầu tiên mở VideoCapture và random seek cho từng window, làm một video có thể mất hơn 2 giờ.

Evaluator sau đó được tối ưu:

- mỗi video decode tuần tự đúng một lần;
- cache frame tensor cần thiết;
- tái sử dụng frame giữa các overlapping windows;
- batch inference = 4;
- tách riêng decode time và model-forward time.

Ví dụ video 002:

- bản cũ: khoảng **8,708 s**;
- bản tối ưu: khoảng **151 s**;
- nhanh hơn xấp xỉ **58 lần**.

---

# 10. CTPGesture continuous RAW result

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

## 10.1. Phân tích isolated → continuous

- Isolated Accuracy: **98.55%**.
- Isolated Macro-F1: **97.96%**.
- Continuous Macro-F1: **80.61%**.
- Continuous Gesture Recall: **94.80%**.
- Temporal Overlap: **65.32%**.

Như vậy model vẫn nhận ra phần lớn gesture thật, nhưng biên thời gian chưa chính xác.

Bottleneck hiện tại nhiều khả năng nằm ở:

- transition giữa gesture và no-gesture;
- transition giữa hai gesture liên tiếp;
- mixed-action windows;
- start/end boundary;
- xác suất bị kéo dài sang frame lân cận;
- cách aggregation từ overlapping windows.

Nói cách khác, **classification đã tốt nhưng temporal localization chưa tốt tương ứng**.

---

# 11. Post-processing trên CTPGesture

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
- Smoothing làm giảm rõ Macro-F1 và Recall.
- MIN_GESTURE = 1.5 s xóa quá nhiều gesture thật.

### Kết luận

Post-processing rule-based chỉ làm sạch nhẹ prediction, **không giải quyết được core temporal problem**.

Do vậy:

- **RAW nên là kết quả continuous chính**.
- R3 chỉ nên dùng như ablation/post-processing result.

---

# 12. Literature trên CTPGesture / Police Gesture Dataset

## 12.1. MD-GCN — Electronics 2021

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

### Cảnh báo so sánh

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

> Đây là **literature context**, không phải fair SOTA leaderboard.

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

### Nhận xét

CTPGesture dễ phân loại isolated hơn rõ rệt vì gesture toàn thân có biên độ lớn và hình thái khác nhau rõ hơn.

Tuy nhiên continuous Frame Accuracy của hai dataset đều quanh 84%.

Điều này cho thấy:

> **Classifier accuracy cao không tự động giải quyết continuous temporal segmentation.**

Đây là vấn đề nghiên cứu quan trọng nhất đang xuất hiện từ các thực nghiệm.

---

# 15. Khả năng real-time hiện tại

## IPN-Hand

- 64-frame window tại 30 FPS = khoảng 2.13 s context.
- step 16 = prediction mỗi khoảng 0.53 s.

## CTPGesture

- 32-frame window tại 15 FPS = khoảng 2.13 s context.
- step 8 = prediction mỗi khoảng 0.53 s.
- yêu cầu xử lý khoảng **1.875 window/s**.

Trong benchmark:

- model-only khoảng **20–22 clips/s**;
- continuous offline pipeline khoảng **8.67 windows/s**.

Về **compute throughput trên Tesla T4**, pipeline đủ nhanh so với tốc độ window sinh ra.

Tuy nhiên chưa thể gọi đây là end-to-end robot latency vì chưa tính:

- camera capture;
- buffering;
- online decode;
- causal waiting time;
- Jetson latency;
- post-processing;
- command emission.

---

# 16. Trạng thái EgoEvGesture

Công việc chuẩn bị dữ liệu đã thực hiện:

- 10 subjects;
- left/right event camera;
- 20 ZIP raw;
- tổng ZIP khoảng **40.433 GB**;
- 38 gesture classes;
- raw event data dạng .npy;
- đã kiểm tra cấu trúc và tạo staging dataset;
- chưa xử lí dedup Subject 5/10 ở bước raw.

Hiện tại chưa train vì gặp giới hạn **Kaggle private dataset quota**:

- canonical dataset cần khoảng 37.656 GB quota;
- tài khoản chỉ còn khoảng 6.865 GB;
- đây là **private storage quota của tài khoản**, không phải scratch disk và không liên quan đến T4 ×2.

Session scratch vẫn có đủ dung lượng, nhưng persistent private dataset quota không đủ.

---

# 17. Những kết quả có thể báo cáo chắc chắn

1. **VideoMamba-Tiny chỉ ~6.96M params** nhưng đạt hiệu năng rất cạnh tranh.
2. Trên IPN-Hand isolated: **84.65%**, cao hơn nhẹ VideoMAEv2 84.59% trong CoboGesture.
3. Trên IPN-Hand continuous: **83.77% Frame Accuracy / 0.5077 overlap** với R3, rất gần CoboGesture VideoMAEv2 **83% / 0.51**.
4. Trên CTPGesture isolated: **98.55% Accuracy / 97.96% Macro-F1**.
5. Trên CTPGesture continuous RAW: **83.98% Frame Accuracy / 80.61% Macro-F1 / 65.32% overlap / 94.80% Recall**.
6. Rule-based post-processing chỉ giúp nhẹ, không giải quyết được boundary problem.
7. Khoảng cách isolated → continuous chứng minh nhu cầu nghiên cứu thêm **temporal modeling/localization**.

---

# 18. Những điểm chưa được phép kết luận quá mức

- Chưa thể nói VideoMamba “SOTA CTPGesture continuous”.
- Chưa có causal evaluation giống RSC-GestureNet.
- Chưa chạy multi-seed.
- Chưa có Segmental F1@10/25/50.
- Chưa có Edit Score.
- Chưa benchmark end-to-end trên Jetson.
- GFLOPs hiện là analytical/custom estimate, cần thống nhất quy ước MAC/FLOP.
- Các bài CTPGesture isolated không dùng cùng split, nên không so thứ hạng tuyệt đối.

---

## B. Khó khăn hiện tại

1. **Giới hạn thời gian Kaggle**
   - CTPGesture training dừng ở 28/30 epochs do session timeout.
   - Best checkpoint vẫn đã lưu thành công.

2. **Continuous recognition còn yếu ở temporal boundary**
   - isolated gần 99%;
   - continuous Macro-F1 chỉ khoảng 80.6%.

3. **Literature không đồng nhất protocol**
   - cùng dataset lineage nhưng test split và metric khác nhau.

4. **EgoEvGesture bị giới hạn storage**
   - private quota chưa đủ để lưu canonical raw dataset ~40 GB.

---

## C. Công việc tiếp theo

### P0 — ưu tiên cao nhất

1. **Triển khai causal evaluator cho CTPGesture**
   - 32-frame causal window;
   - stride 4;
   - end-aligned prediction;
   - tuyệt đối không dùng future frame;
   - report Accuracy, Macro-F1, Online-F1;
   - so cùng protocol với RSC-GestureNet.

2. **Error/keyframe visualization**
   - tạo file errors.csv/txt;
   - lưu video, frame, GT, prediction, confidence;
   - show keyframe/window sai;
   - nhóm lỗi theo transition và gesture pair.

3. **Tách validation cho post-processing**
   - chọn threshold trên validation;
   - không tune trực tiếp trên official test.

### P1 — publication-quality experiments

4. Chạy ít nhất 3 random seeds.
5. Bổ sung Segmental F1@10/25/50.
6. Bổ sung Edit Score.
7. Phân tích calibration/confidence ở transition.
8. Đo peak VRAM và latency chuẩn.
9. Xác minh lại FLOP convention.

### P2 — EgoEvGesture

10. Giải quyết Kaggle storage hoặc đọc trực tiếp từ nguồn ngoài.
11. Kiểm toán số sequence thực tế.
12. Xây dựng event representation.
13. Train VideoMamba baseline.
14. Đánh giá cross-subject / cross-camera.

---

# D. Kết luận ngắn để trình bày

> Trong giai đoạn này em đã hoàn thiện được pipeline đánh giá VideoMamba-Tiny từ isolated tới continuous trên IPN-Hand và CTPGesture v1. Trên IPN-Hand, VideoMamba-Tiny đạt 84.65% isolated accuracy và khoảng 83.77% continuous frame accuracy với overlap 0.508, trong khi chỉ có khoảng 6.96M parameters, thấp hơn khoảng 43 lần so với VideoMAEv2 mà bài CoboGesture sử dụng. Trên CTPGesture, isolated accuracy đạt 98.55%, nhưng khi chuyển sang full-video sliding-window thì frame accuracy giảm xuống 83.98% và Macro-F1 còn 80.61%. Gesture Recall vẫn cao 94.80%, cho thấy vấn đề chính hiện tại nằm ở temporal boundary/localization hơn là khả năng phân biệt gesture class. Em đã thử các bước post-processing nhưng mức cải thiện nhỏ, do đó hướng tiếp theo em đề xuất là xây dựng causal evaluation 32/4 giống RSC-GestureNet và tập trung vào temporal modeling thay vì tiếp tục tối ưu heuristic hậu xử lí.

---

# E. Tài liệu tham khảo chính

1. G. Benitez-Garcia et al., **IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition**, ICPR.
2. P.-D. Nguyen et al., **CoboGesture: A Continuous Hand Gesture Dataset and Recognition Method for Human-Collaborative Robot Interaction**, IEEE Access, vol. 14, pp. 27584–27601, 2026. DOI: 10.1109/ACCESS.2026.3664027.
3. X. Xiong et al., **Traffic Police Gesture Recognition Based on Gesture Skeleton Extractor and Multichannel Dilated Graph Convolution Network**, Electronics, 2021.
4. Xiao et al., **A Lightweight and Efficient Gesture Recognizer for Traffic Police Commands Using Spatiotemporal Feature Fusion**, Scientific Reports, 2025.
5. Zhang et al., **A Traffic Police Gesture Recognition Method Based on BiLSTM-Transformer Architecture**, Electronics, 2026.
6. Li et al., **RSC-GestureNet: Reliability-Aware Selective Causal Recognition of Chinese Traffic Police Gestures**, arXiv:2608.02200v1, 2026.

## Notebook / report nguồn trong repository

- Research_Report/260921
- Research_Report/261007/videomamba_man_configsBenchmark.ipynb
- Research_Report/261015/VideoMambaTinyCTPGestureFinal.ipynb
- Research_Report/261015/Readme.md
- Research_Report/261015/CTPGesture_VideoMamba_Report_20261015.pdf
