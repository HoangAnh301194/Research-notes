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


### 1.4. Computational complexity và runtime benchmark (bổ sung từ notebook)

**Nguồn gốc số liệu:** `VideoMambaTinyCTPGestureFinal.ipynb`, Cell 8 (checkpoint), Cell 22 (continuous timing), Cell 26 (complexity và benchmark FP32); không phải số đo lấy từ các bài báo đối chứng.

| Thông số | Kết quả | Phạm vi đo / điều kiện |
|---|---:|---|
| Model | VideoMamba-Tiny | 24 layers, 3.137 tokens/clip |
| Input model | 1 × 3 × 16 × 224 × 224 | RGB, FP32 |
| Parameters | **6.959.817 (6,96M)** | Đếm tất cả model parameters |
| Parameter storage (FP32) | 26,55 MiB | Chỉ trọng số, không tính activation |
| **Estimated computation** | **29,026 GFLOPs/clip** | Counter thủ công ở Cell 26; cần xác minh quy ước MAC/FLOP khi so với bài khác |
| Checkpoint file size | 80,14 MiB | File .pth chứa cấu trúc lưu checkpoint; không đồng nhất với trọng số FP32 |
| Mean/median/p95 latency | **50,65 / 50,79 / 51,00 ms/clip** | Batch=1; 5 warm-up + 20 runs; CUDA events, native_cuda, Tesla T4 |
| Model throughput | **19,74 clips/s** | 1000/mean_ms, model-only, batch=1 |
| Additional peak GPU allocation | 56,58 MiB | Cấp phát tăng thêm trong forward; **không phải** tổng peak VRAM |
| Continuous mean model latency | **48,17 ms/window** | Full-test loop, batch=4; khác phép đo batch=1 |
| Continuous mean wall time | **115,35 ms/window** | Bao gồm decoding, preprocessing, batching và overhead đánh giá |
| Effective processing throughput | **8,67 windows/s** | 8.392 windows / 968,01 s, full-test loop |
| Total decode/preprocessing | 495,91 s | 11 test videos |
| Total model forward | 404,23 s | 11 test videos |
| Total wall time | 968,01 s | 11 test videos, không phải streaming camera |

**Lưu ý về FLOPs:** con số **17,0 GFLOPs** trong bản ghi chú trước không khớp Cell 26. Trong lần chạy này, custom counter cho ra **29,026 GFLOPs**. Counter dựa trên phép ước lượng riêng cho Conv3D, projections và selective scan (không phải profiler phần cứng chuẩn hóa); cần kiểm toán quy ước multiply–accumulate và các toán tử bỏ sót trước khi so sánh GFLOPs với các bài khác.

**Lưu ý về realtime:** 19,74 clips/s không đồng nghĩa camera FPS. Pipeline continuous đọc cửa sổ 32 frame (~2,13 s video tại 15 FPS), stride=8 frame (~0,533 s), sau đó phân bổ xác suất cửa sổ lên các frame bên trong. Pipeline này có thể sử dụng thông tin frame tương lai để gán nhãn các frame đầu cửa sổ. Kết quả xử lý nhanh hơn tốc độ đến trung bình của các cửa sổ trên Tesla T4 **chưa chứng minh** hệ thống online-causal, chưa chứng minh độ trễ đưa ra lệnh tại camera, và chưa phải benchmark trên thiết bị edge.

**Tình trạng thống kê:** Cell 26 chỉ đo một checkpoint trên một môi trường, chưa có mean±std qua nhiều seed, total peak VRAM, end-to-end streaming latency, Edit Score và Segmental F1@10/25/50. Các phép đo này chưa được phép điền số giả định vào bảng so sánh.

**Tình trạng post-processing:** R3 (83,99% FrameAcc, 80,63% Macro-F1) là cấu hình **được chọn sau khi thử trực tiếp trên test set**, do đó chỉ là exploratory analysis. Kết quả chính để báo cáo hiện nay là **RAW: 83,98% Frame Accuracy, 80,61% Macro-F1**. Để báo cáo R3 như kết quả độc lập, cần chọn/tối ưu siêu tham số trên validation trước khi mở test.



### 2. Các nghiên cứu đối chứng trên CTPGesture v1 / Police Gesture Dataset

**Phạm vi:** Bốn bài PDF đã lưu trong thư mục này. Các bài có cùng *họ dữ liệu* nhưng có thể khác split, đầu vào, đơn vị dự đoán và metric; bảng bên dưới là **literature comparison**, KHÔNG phải bảng chạy lại chung protocol. Bài Wu et al. (AMSTCNN) không có PDF nên không nằm trong báo cáo này.

#### 2.1. Thông tin công bố, venue và xếp hạng

| Mã | Bài báo, năm | Venue | Xếp hạng có ghi rõ hệ và năm | PDF tại thư mục này |
|---|---|---|---|---|
| [P1] | Xiong et al., **2021**, *Traffic Police Gesture Recognition Based on Gesture Skeleton Extractor and Multichannel Dilated Graph Convolution Network* | **Electronics** (MDPI), 10(5):551 | **JCR 2021: Q3**; JCR 2025: **Q2** (ngành Engineering, Electrical and Electronic). SJR 2021: Q2 | [electronics-10-00551-v2.pdf](electronics-10-00551-v2.pdf) |
| [P2] | Xiao, Li & Zhao, **2025**, *A Lightweight and Efficient Gesture Recognizer for Traffic Police Commands Using Spatiotemporal Feature Fusion* | **Scientific Reports** (Nature Portfolio), 15:18256 | **JCR 2025: Q1** (Multidisciplinary Sciences) | [s41598-025-02833-y.pdf](s41598-025-02833-y.pdf) |
| [P3] | Zhang et al., **2026**, *A Traffic Police Gesture Recognition Method Based on BiLSTM-Transformer Architecture* | **Electronics** (MDPI), 15(12):2578 | **JCR 2025: Q2** (ngành Engineering, Electrical and Electronic; dùng dữ liệu JCR hiện có trước công bố 2026) | [electronics-15-02578.pdf](electronics-15-02578.pdf) |
| [P4] | Li, Gao & Fu, **2026**, *RSC-GestureNet: Reliability-Aware Selective Causal Recognition of Chinese Traffic Police Gestures* | **arXiv:2608.02200v1** | **Preprint**, không được gán JCR quartile/CORE; thông tin về proceedings cần kiểm chứng độc lập | [2608.02200v1.pdf](2608.02200v1.pdf) |

**Nguồn metadata/ranking:**
- [P1 DOI](https://doi.org/10.3390/electronics10050551); [P2 DOI](https://doi.org/10.1038/s41598-025-02833-y); [P3 DOI](https://doi.org/10.3390/electronics15122578); [P4 arXiv](https://arxiv.org/abs/2608.02200).
- [MDPI Electronics — Journal Statistics](https://www.mdpi.com/journal/electronics/stats) công bố nhiều JCR quartile **theo ngành**; [bảng lịch sử Electronics JCR/SJR](https://www.iit.comillas.edu/publicacion/info_revista/en/530/Electronics) giúp đối chiếu năm 2021 và 2025.
- [Scientific Reports — JCR quartile history](https://www.akaturk.com/journals/21100200805?lang=en), category Multidisciplinary Sciences. Xếp hạng venue chỉ mang tính bối cảnh; **không đồng nghĩa bài có citation impact cao**.

#### 2.2. Phương pháp, đóng góp và bằng chứng trích dẫn trong từng bài

**[P1] Xiong et al. (2021) — GSE + MD-GCN.** Tác giả dùng Gesture Skeleton Extractor để giảm các khớp không hữu ích, tiếp đó đưa đặc trưng pose vào Multichannel Dilated Graph Convolution Network với nhiều dilation rates, học quan hệ không gian–thời gian. Bài chính là skeleton/GCN, nhưng có bảng đối chứng quan trọng với các model nhận dạng RGB video. **Section 6.1, trang in 9/14**: tác giả mô tả 20 video; chia **11 train (1.789 gesture clips), 9 test (1.565 gesture clips)** và cắt clip dựa trên annotation. **Section 6.3, Table 3, trang in 12/14**: Hara/3D ResNet **81,16%**, Tran/C3D **87,12%**, Qiu/P3D **92,21%**, phương pháp đề xuất **98,95%**. Các giá trị của Hara/Tran/Qiu là **re-evaluation được báo cáo trong P1**, không phải kết quả CTPGesture được công bố trong các bài kiến trúc nguyên gốc. [Nguồn P1](https://www.mdpi.com/2079-9292/10/5/551).

**[P2] Xiao et al. (2025) — NTPGR.** EPFFNet trích xuất keypoints, xây dựng đặc trưng góc/khoảng cách, MSNet học thời gian bằng ba nhánh **CNN + LSTM sequence + temporal/channel attention**. **Abstract; Fig. 1; Methods, trang in 1–4/19** là bằng chứng kiến trúc. **Table 2, trang in 11/19**: test **10 video**, **778 gesture instances**, **759 nhận dạng đúng**, **97,56% Accuracy**. **Table 3, trang in 13/19** đối chiếu sequence/composite baselines. **Table 4 và đoạn phân tích trang in 14/19** báo cáo **0,76 s average response time** trên GTX TITAN XP, không phải latency model-only trên T4. **Experimental Settings** mô tả dữ liệu là “21 short videos” nên có sai khác metadata so với nguồn 20-video và notebook — cần ghi là **related dataset**, chưa chứng minh hoàn toàn cùng cách chia. [Nguồn P2](https://www.nature.com/articles/s41598-025-02833-y).

**[P3] Zhang et al. (2026) — YOLOv11m-Pose + BiLSTM–Transformer.** Tác giả trích **17 keypoints/frame**, tạo chuỗi **40 frame**; BiLSTM nắm quan hệ cục bộ và Transformer self-attention học phụ thuộc xa. **Section 3, Fig. 5** là bằng chứng kiến trúc. **Section 4, Table 1, trang in 13/23**: liệt kê **10 test videos**, gồm 002, 004, 008, 010, 012, 014, 016, 018, 102, 104; **thiếu video 101** có trong notebook của nhóm. **Table 2, trang in 16/23**: cấu hình YOLOv11m-Pose + hybrid đạt **98,91% Accuracy/F1**; **Table 3, trang in 17/23**: BiLSTM **96,48%**, Transformer **91,24%**, hybrid **98,91%**, hybrid recognition-head **86,41K parameters và 1,3299 s/sequence**; **Table 4, trang in 19/23** so sánh với các công trình khác. **Chú ý lỗi nhất quán:** đoạn diễn giải Table 3 lại gọi standalone BiLSTM là 98,91%, không trùng số Table 3 96,48%; sử dụng **số trong bảng** và ghi nhận sự không nhất quán. Latency **không tính YOLO pose extraction**, không so trực tiếp với RGB model full inference. Bài mô tả ảnh 1024×768, trong khi notebook của nhóm kiểm tra video 1080×1080. [Nguồn P3](https://www.mdpi.com/2079-9292/15/12/2578).

**[P4] Li et al. (2026) — RSC-GestureNet.** Dùng skeleton features và pose–RGB late fusion, reliability-aware gating, temporal causal dilated convolutions, calibration và stable command emission. **Section 3.4–3.5, trang in 7–8/14** mô tả kiến trúc; **Section 4.1, trang in 8/14** nêu **official CTPGesture v1 split, 9 classes, 32-frame causal window, stride 4, 3 random seeds**, không dùng test để chọn threshold. **Table 1, trang in 9/14**: RSC-GestureNet **93,33±0,24% Accuracy**, **91,71±0,27% Macro-F1**, **91,69±0,29% Online-F1**, **TTC 0,153±0,013 s**. Cũng ở **Table 1**, các đối chứng do nhóm tác giả tái triển khai gồm GRU **88,82/87,10% (Acc/Macro-F1)**, TCN **89,88/88,29%**, MD-GCN† **90,08/88,48%**. **Chú ý:** MD-GCN† **90,08%** ở P4 là reimplementation trong giao thức causal, **không phải** MD-GCN **98,95%** trong P1. [Nguồn P4](https://arxiv.org/pdf/2608.02200).

#### 2.3. Bảng provenance: chỉ số chính xác trích từ đâu?

| Model / nghiên cứu | Dataset được tác giả ghi | Published result | Đơn vị đánh giá | Vị trí bằng chứng |
|---|---|---|---|---|
| 3D ResNet [P1, Hara baseline] | Police Gesture Dataset | Acc **81,16%** | Gesture clip | P1, **Table 3**, PDF trang in **12** |
| C3D [P1, Tran baseline] | Police Gesture Dataset | Acc **87,12%** | Gesture clip | P1, **Table 3**, trang **12** |
| P3D [P1, Qiu baseline] | Police Gesture Dataset | Acc **92,21%** | Gesture clip | P1, **Table 3**, trang **12** |
| MD-GCN [P1] | Police Gesture Dataset | Acc **98,95%** | Gesture clip | P1, **Table 3**, trang **12** |
| NTPGR [P2] | Police Gesture Dataset | Acc **97,56%** = 759/778 | Gesture instance | P2, **Table 2**, trang **11** |
| BiLSTM–Transformer [P3] | Police Gesture Dataset từ nguồn gốc, được xử lý lại | Acc/F1 **98,91%** | Temporal sequence prediction | P3, **Tables 2–4**, trang **16–19** |
| RSC-GestureNet [P4] | **CTPGesture v1 official split** | Acc **93,33±0,24%**, Macro-F1 **91,71±0,27%** | Full-frame causal | P4, **Table 1**, trang **9** |
| VideoMamba-Tiny (nhóm) | **CTPGesture v1**, 11 test videos | Isolated Acc **98,55%** (9 classes); RAW FrameAcc **83,98%**, Macro-F1 **80,61%** | Isolated / continuous (tách riêng) | Notebook **Cells 10, 16, 22** |

**Quan trọng:** Accuracy trong các bảng của P1/P2/P3/P4 **không được tự đổi tên thành Edit Score hoặc “Edit Distance Accuracy”**. Nếu một bài dùng định nghĩa metric đặc biệt, cần trích phương trình gốc trước khi thống nhất tên. VideoMamba 98,55% là clip-level 9 classes, còn P1 chủ yếu báo cáo gesture clips đã cắt; ngay cả khi cùng tên Accuracy cũng chưa chắc cùng mẫu và cách chia.

#### 2.4. So sánh VideoMamba-Tiny: tách isolated và continuous

**Bảng A — Published gesture/clip recognition results (different protocols; KHÔNG phải fair benchmark):**

| Model | Modality | Kết quả tác giả công bố | VideoMamba-Tiny (tham khảo) | Điều kiện khác biệt chính |
|---|---|---:|---:|---|
| 3D ResNet, P1 re-evaluation | RGB clip | 81,16% Acc | — | P1 dùng 9 test videos; clipped gestures |
| C3D, P1 re-evaluation | RGB clip | 87,12% Acc | — | Khác split |
| P3D, P1 re-evaluation | RGB clip | 92,21% Acc | — | Khác split |
| MD-GCN, P1 | Skeleton | 98,95% Acc | — | Khác modality và split |
| NTPGR, P2 | Skeleton temporal | 97,56% Acc | — | 10 test videos / 778 instances |
| BiLSTM–Transformer, P3 | Pose + temporal | 98,91% Acc | — | 10 test videos; 40-frame sequences |
| **VideoMamba-Tiny (ours)** | **RGB video** | — | **98,55% isolated Accuracy (1.728, 9 classes)** | Ground-truth boundaries; 11 test videos |
| **VideoMamba-Tiny, gesture-only** | RGB video | — | **98,02% Accuracy (859, GT classes 1–8)** | Ground-truth boundaries; 11 test videos |

**Bảng B — Published continuous recognition (different protocols; KHÔNG phải fair causal benchmark):**

| Model | Accuracy (%) | Macro-F1 (%) | Giao thức |
|---|---:|---:|---|
| GRU, tái triển khai trong P4 | 88,82±0,43 | 87,10±0,45 | P4 Table 1, causal 32/4 |
| TCN, tái triển khai trong P4 | 89,88±0,54 | 88,29±0,62 | P4 Table 1, causal 32/4 |
| MD-GCN†, tái triển khai trong P4 | 90,08±1,38 | 88,48±1,43 | P4 Table 1, causal 32/4 |
| RSC-GestureNet, P4 | **93,33±0,24** | **91,71±0,27** | P4 Table 1, official split, causal 32/4 |
| **VideoMamba-Tiny RAW (ours)** | **83,98** | **80,61** | 11 test videos; window 32/stride 8, overlapping probability aggregation; 1 checkpoint |
| VideoMamba-Tiny R3 (exploratory) | 83,99 | 80,63 | Post-processing selected on test; **không dùng làm điểm SOTA chính** |

**Sự khác biệt cốt lõi:** VideoMamba khi phân loại isolated sử dụng ground-truth boundaries; khi continuous lại ánh xạ xác suất của cửa sổ chồng lấn lên các frame. Với frame ở đầu cửa sổ, phép gán này **dựa vào frame tương lai**; **không cùng định nghĩa causal** của P4. Khác biệt seed, batch size, feature extraction, split và quy tắc dự đoán khiến việc lấy chênh lệch số học để tuyên bố “outperform SOTA” là không hợp lệ.

#### 2.5. Tính tương thích, thảo luận và kế hoạch kiểm chứng

| Nghiên cứu | Split test công bố | Unit/metric | So sánh trực tiếp với notebook? | Việc cần làm nếu muốn fair comparison |
|---|---|---|---|---|
| P1 — MD-GCN/C3D/P3D | **9 videos, 1.565 gesture clips** | Clip Accuracy | **Không** | Chạy các baseline RGB trên đúng 11 test videos + cùng segmentation protocol, hoặc đánh giá VideoMamba trên đúng 9 test videos của P1 |
| P2 — NTPGR | **10 videos, 778 gestures**, không có 101 | Gesture-instance Accuracy | **Không** | Lọc VideoMamba về đúng 10 video và chuẩn hóa định nghĩa recognition success; đối chiếu 778 gesture instances |
| P3 — BiLSTM–Transformer | **10 videos**, không có 101 | Sequence prediction Accuracy/F1 | **Không** | Chuẩn hóa train/test và vị trí gán nhãn trong cửa sổ; báo cáo cả pose extractor cost |
| P4 — RSC-GestureNet | **Official v1**, 9 classes, causal 32/4 | Frame Accuracy, Macro-F1, Online-F1, TTC | **Gần nhất về continuous**, nhưng **chưa trực tiếp** | Chạy VideoMamba với 32/4 **causal end-aligned**, không dùng future frames, cùng metric, seed và preprocessing |

**Các câu kết luận có thể bảo vệ:**
- VideoMamba-Tiny có **6,96M parameters**, nhận trực tiếp RGB, đạt **98,55% isolated Accuracy** và **83,98%/80,61% continuous RAW FrameAcc/Macro-F1** trên **11 video test** đã audit.
- Kết quả isolated cao **không suy ra** khả năng phát lệnh online: tập isolated dùng GT boundaries, còn continuous phải tự giải quyết biên cử chỉ.
- Tính ưu việt về số tham số hoặc latency so với các mô hình pose **chưa chứng minh**, vì nhiều bài chỉ báo cáo temporal head và dùng GPU khác.
- P4 cung cấp protocol gần nhất về nhận dạng liên tục; mức 93,33/91,71% chỉ là **published reference**, chưa phải so sánh chung giao thức.
- Hiện chưa thể gọi cấu hình VideoMamba-Tiny này là “phương pháp cải tiến kiến trúc mới” nếu chưa chỉ ra và ablate thay đổi so với backbone VideoMamba gốc.

**Các việc cần đo/chuẩn hóa trước khi nộp bài khoa học:** (i) lập validation split nội bộ từ train; không tune hậu xử lý trên official test, (ii) chạy ≥3 random seeds, (iii) thêm Edit Score và Segmental F1@10/25/50, (iv) benchmark model-only và full streaming pipeline có độ trễ đầu-cuối, (v) verify GFLOPs theo một quy ước thống nhất với baselines, (vi) mô tả rõ training config, pretraining weights, checkpoint policy, (vii) re-run P3D/C3D và RSC-like causal baselines cùng split nếu đưa vào bảng fair comparison.

#### 2.6. Danh mục PDF đối chứng

1. [P1 — Xiong et al., Electronics 2021](electronics-10-00551-v2.pdf): Section 6.1 trang 9; **Table 3 trang 12**.
2. [P2 — Xiao et al., Scientific Reports 2025](s41598-025-02833-y.pdf): **Tables 2–4 trang 11–14**.
3. [P3 — Zhang et al., Electronics 2026](electronics-15-02578.pdf): **Table 1 trang 13; Tables 2–4 trang 16–19**.
4. [P4 — Li et al., arXiv 2026](2608.02200v1.pdf): **Section 4.1 trang 8; Table 1 trang 9**.



## B. Khó khăn 
- Dataset CTPGesture v1 có thời lượng video khá dài ở 15 FPS, việc đánh giá toàn diện Continuous Sliding Window trên toàn bộ 11 test video đòi hỏi tài nguyên tính toán lớn.
- Quá trình training các dataset lớn hơn (như CTPGesture v2 và EgoEvGesture) trên Kaggle thường chạm ngưỡng giới hạn 12 tiếng runtime, cần giải pháp lưu checkpoint và resume training định kỳ.

## C. Công việc tiếp theo 
- Tiếp tục nghiên cứu các giải pháp tối ưu temporal window cho Continuous recognition đối với các gesture chuyển động chậm.
- Chuẩn bị dữ liệu và triển khai huấn luyện trên dataset **EgoEvGesture** (dữ liệu cử chỉ từ event camera góc nhìn thứ nhất) và **CTPGesture v2** (33 lớp gồm cử chỉ kết hợp hướng cơ thể).
- Báo cáo kết quả và xin ý kiến định hướng nghiên cứu tiếp theo từ Thầy.
