# Báo cáo nghiên cứu ngày 22/10/2026

## A. Công việc đã làm 
- Tổng quan lại toàn bộ kết quả đã làm 
- Khảo sát các bài báo :
    - Cùng sử dụng IPN Hand : none Skeleton 
    - Cùng sử dụng CTP Gesture v1 : 1-2 bài SOTA có skeleton và ~5 bài RGB input 


## B. Khó khăn 

## C. Công việc tiếp theo


**Bằng chứng ký hiệu trong bảng:** `[O]` số đo/output từ notebook nhóm; `[P]` số được bài báo/benchmark gốc công bố; `[E]` số ước lượng kiến trúc hoặc đo lại bằng code nhóm (không phải số do công trình đối chứng công bố); `—` không thấy số có thể xác nhận.

**Mức độ đối chiếu:**
- **A:** Cùng dataset và metric tương đồng, còn phải xác nhận chi tiết split/preprocessing trước khi gọi là fair comparison.
- **B:** Cùng tên dataset nhưng khác input, subset, split, aggregation hoặc test protocol; chỉ đối chiếu tham khảo.
- **C:** Khác định nghĩa tác vụ/metric, hoặc phương pháp chỉ là nghiên cứu liên quan; **không so sánh thứ hạng**.

### Bảng 1 — IPN-Hand: Isolated Gesture Recognition (bản hoàn thiện dữ liệu)

**Giao thức:** Các phương pháp dùng kết quả isolated clip trên IPN-Hand; các baseline gốc sử dụng 32-frame RGB/RGB-Flow/RGB-Seg, VideoMamba nhóm dùng 16×224². Cột latency là **model / pipeline inference time do từng nghiên cứu tự đo**, ghi thiết bị; **không** so trực tiếp T4 với GTX 1080 Ti. `†` là *GFLOPs của kiến trúc đo bởi nhóm* chứ không phải từ paper Cobo. `—` là **chưa có số đúng biến thể**.

| Method | Năm | Input | **Params classifier (M)** | **Params pipeline (M)** | GFLOPs/clip | **Accuracy (%)** | **Latency (ms), GPU** | Nguồn |
|---|---:|---|---:|---:|---:|---:|---|---|
| **VideoMamba-Tiny (OURS)** | 2026 | RGB 16×224² | **6.9608** | **6.9608** | **29.0261†** | **84.65** | **48.48, T4†** | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAEv2 | 2026 | RGB ViT-L 16×224²† | **303.8** | **303.8** | **596.8327†** | 84.59 | **459.3, T4†** | [Cobo Table 4](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction) · [benchmark kiến trúc nhóm](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAE | 2026 | RGB ViT-L 16×224²† | **303.8** | **303.8** | **596.8327†** | 78.81 | **484.9, T4†** | [Cobo Table 4](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction) · [benchmark kiến trúc nhóm](../261007/videomamba_man_configsBenchmark.ipynb) |
| ResNeXt-101 RGB+Flow | 2020/21 | RGB + SPyNet, 32f | 47.56 | **49.000*** | — | 86.32 | **50.8, GTX1080Ti** | [IPN Table IV/V](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 RGB+Seg | 2020/21 | RGB + HarDNet, 32f | 47.56 | **51.674*** | — | 84.77 | **37.4, GTX1080Ti** | [IPN Table IV/V](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 RGB | 2020/21 | RGB, 32f | 47.51 | 47.51 | — | 83.59 | **27.7, GTX1080Ti** | [IPN Table IV](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| C3D RGB | 2020/21 | RGB, 32f | 50.75 | 50.75 | — | 77.75 | **76.2, GTX1080Ti** | [IPN Table IV](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 RGB+Seg | 2020/21 | RGB + HarDNet, 32f | 46.27 | **50.384*** | — | 75.11 | **25.9, GTX1080Ti** | [IPN Table IV/V](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 RGB+Flow | 2020/21 | RGB + SPyNet, 32f | 46.27 | **47.710*** | — | 74.65 | **39.9, GTX1080Ti** | [IPN Table IV/V](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 RGB | 2020/21 | RGB, 32f | 46.25 | 46.25 | — | 73.10 | **18.2, GTX1080Ti** | [IPN Table IV](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) |

**Chú thích bắt buộc:** `*` Params pipeline = classifier + SPyNet (**1.440M**) hoặc HarDNet (**4.114M**), lấy thông số các module từ Table V của IPN-Hand; tổng là **suy luận theo tổ hợp module**, không phải tác giả trực tiếp công bố. `†` GFLOPs / latency VideoMAE v1/v2 và VideoMamba đều là **benchmark kiến trúc của notebook nhóm** (input 16×224², FP32, T4, định nghĩa 1 MAC = 1 FLOP-equivalent), không phải FLOPs/latency trên IPN-Hand của nhóm CoboGesture. Các tác giả IPN gốc không công bố GFLOPs đúng biến thể; để trống có chủ đích. Kết quả 84.65% của VideoMamba dùng 1.101 isolated clips, loại class no-gesture; phải khớp chi tiết protocol trước khi so xác suất khác biệt 0.06 điểm phần trăm với Cobo.


### Bảng 2 — IPN-Hand: Continuous Gesture Recognition


#### Bảng 2A — So sánh theo frame và đoạn cử chỉ (CoboGesture 2026)

| Method | Input | **Params recognition (M)** | **Params pipeline (M)** | **Mean FrameAcc (%)** | **Activity Overlap** | **Gesture Recall (%)** | Nguồn |
|---|---|---:|---:|---:|---:|---:|---|
| **VideoMamba Raw (OURS)** | RGB, offline | **6.9608** | **6.9608** | **83.4889** | **0.491110** | **78.8995** | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| VideoMamba R3 (test-tuned) | RGB, offline | 6.9608 | 6.9608 | 83.5249 | 0.507705 | 76.2844 | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAEv2 | RGB, offline | 303.8 | 303.8 | **83.00** | **0.510** | — | [Cobo Table 7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction) |
| RGB-CoGes + VideoMAE | RGB, offline | 303.8 | 303.8 | 80.00 | 0.370 | — | [Cobo Table 7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction) |
| DD-Net continuous | Skeleton | **2.4†** | — (pose extractor) | 57.00 | — | — | [Cobo Table 3/7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction) |

`†` 2.4M là Params DD-Net **trong thí nghiệm CoboGesture (Table 3)**; chưa chứng minh bản chạy IPN Table 7 dùng đúng cấu hình này. Cobo không công bố Gesture Recall theo định nghĩa riêng trong notebook của nhóm nên không thể điền.

#### Bảng 2B — IPN-Hand official Levenshtein benchmark (model detector + classifier)

| Method | Input | **Params classifier (M)** | **Params pipeline (M)** | **Levenshtein Accuracy (%)** | **Inference (ms, GTX1080Ti)** | Nguồn |
|---|---|---:|---:|---:|---:|---|
| ResNeXt-101 + ResLight + SPyNet | RGB+Flow | 47.56 | **49.908*** | **42.47** | **53.7** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight + HarDNet | RGB+Seg | 47.56 | **52.582*** | **39.01** | **39.9** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight | RGB | 47.51 | **48.405*** | **25.34** | **30.1** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 + ResLight + SPyNet | RGB+Flow | 46.27 | **48.618*** | **39.47** | **43.1** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 + ResLight + HarDNet | RGB+Seg | 46.27 | **51.292*** | **33.27** | **29.2** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNet-50 + ResLight | RGB | 46.25 | **47.145*** | **19.78** | **20.4** | [IPN Table IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) · [leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |


### Bảng 3 — CTPGesture v1: Isolated / clip classification

**Giao thức VideoMamba nhóm:** 11 video test/1.728 segments, class 0–8, uniformly sample 16 RGB frames. **Các baseline Xiong (2021) dùng 9 video test/1.565 segments; không xếp hạng trực tiếp**. Khác IPN, nhiều công trình CTP không công bố Macro-F1 và Params đúng biến thể, nên bảng này ưu tiên **Accuracy + complexity có bằng chứng**; Macro-F1 riêng của nhóm giữ trong chú thích, không lấp bằng giá trị không cùng protocol.

| Method | Năm | Input | **Params recognizer (M)** | **Params full pipeline (M)** | **GFLOPs/clip** | **Accuracy (%)** | Bằng chứng/thời gian | Nguồn |
|---|---:|---|---:|---:|---:|---:|---|---|
| **VideoMamba-Tiny (OURS, 9 lớp)** | 2026 | RGB, 16×224² | **6.959817** | **6.959817** | **29.0261†** | **98.5532** | Macro-F1=**97.9590%**; 1.728 clips | [CTP notebook cell 16/26](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| GSE + MD-GCN (Xiong) | 2021 | OpenPose→skeleton, 80f | **—** | **—** | — | **98.95** | Model Size = **11.7 M** (*đơn vị chưa xác thực*); Inference Time = **3.0 s** (theo Table 1, đơn vị đo thao tác chưa rõ) | [Xiong Table 1/3](https://www.mdpi.com/2079-9292/10/5/551) |
| P3D (Qiu; đối chứng Xiong) | 2021 | RGB | **≈65.3‡** | — | — | **92.21** | P3D-199 reference, **không xác minh checkpoint Xiong** | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [P3D gốc](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf) |
| C3D (Tran; đối chứng Xiong) | 2021 | RGB | **78.4‡** | — | **38.5‡** | **87.12** | C3D MMAction2: **16×112²**, model family reference | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [C3D model zoo](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md) |
| 3D ResNet (Hara; đối chứng Xiong) | 2021 | RGB | **33.371472‡** | — | **40.697‡** | **81.16** | Torchvision R3D-18: **16×112²**, reference | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [R3D-18 model zoo](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html) |

#### Nghiên cứu 2025–2026 cùng chủ đề, nhưng **không phải isolated benchmark tương đương**

| Method | Năm | Input | **Params** | **Result được tác giả công bố** | **Runtime công bố** | Lý do không xếp với VideoMamba | Nguồn |
|---|---:|---|---|---|---|---|---|
| YOLO11m-Pose → BiLSTM–Transformer | 2026 | RGB→Pose | **0.08641M recognition head; ≈20.99M pipeline‡** | **98.91% sequence recognition** | **1.3299 s/sequence head** (chưa tính pose extraction) | 10 test videos, sequence-to-one | [Zhang 2026 Table 2/3](https://www.mdpi.com/2079-9292/15/12/2578) |
| NTPGR (EPFFNet + MSNet) | 2025 | RGB→Pose | **—** (thiếu exact checkpoint) | **97.56% gesture-instance** | **0.76 s average response** trên TITAN XP | Khác split/đơn vị đánh giá | [Xiao 2025](https://www.nature.com/articles/s41598-025-02833-y) |




### Bảng 4 — CTPGesture v1: Continuous Gesture Recognition

**Phát hiện sửa sai khi kiểm tra paper gốc:** RSC-GestureNet-**Fusion** (Accuracy **91.38%**) và RSC-GestureNet-**Ours** (Accuracy **93.33%**) là **hai hàng riêng** ở Table 1, không được gộp! `RSC-GestureNet-Fusion` dùng kiến trúc 0.815270M params recognizer từ code; cấu hình đầy đủ của `RSC-GestureNet (Ours)` gồm calibration/reliability-based branching và sử dụng nhiều nguồn dự đoán, nên cần kiểm tra chính xác full component count. Cả hai đều **chưa gồm MediaPipe/RGB-to-pose extraction**.

| Method | Protocol / Input | **Params recognition (M)** | **Accuracy (%)** | **Macro-F1 (%)** | **Online-F1 (%)** | **Early@10 (%)** | **TTC (s)** | **Robust-F1 (%)** | Nguồn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **VideoMamba Raw (OURS)** | RGB, offline 32/8 | **6.959817** | **83.9836** | **80.6135** | — | — | — | — | [CTP notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| VideoMamba R3 (test-tuned) | RGB, offline 32/8 | 6.959817 | 83.9940 | 80.6254 | — | — | — | — | [CTP notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| RSC-GestureNet **(Ours)** | Pose+RGB, **causal**, calibrated | **≥0.815270*** | **93.33±0.24** | **91.71±0.27** | **91.69±0.29** | **98.80±0.07** | **0.153±0.013** | **90.97±0.37** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [source](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) |
| RSC-GestureNet-**Fusion** | Pose+RGB, causal | **0.815270 [C]** | **91.38±0.38** | **89.95±0.42** | **89.28±0.53** | **98.72±0.20** | **0.191±0.040** | **88.93±0.77** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [source](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) |
| RSC-Gesture-Pose+VC | Pose, causal, calibrated | **≈0.335531†** | **93.09±0.26** | **91.35±0.43** | **91.33±0.39** | **98.41±0.34** | **0.220±0.060** | **89.94±0.26** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [source](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) |
| TCN+VC | Pose, causal, calibrated | — | **92.97±0.18** | **91.16±0.21** | **91.08±0.23** | **98.41±0.07** | **0.214±0.007** | **90.29±0.34** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) |
| ST-GCN+VC | Pose, causal, calibrated | — | **92.47±0.39** | **90.73±0.50** | **90.63±0.57** | **98.21±0.18** | **0.229±0.034** | **87.58±0.79** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) |
| MD-GCN-style reimplemented | Pose, causal | **0.399993 [C]** | **90.08±1.38** | **88.48±1.43** | **89.54±0.76** | **98.14±0.42** | **0.257±0.035** | — | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [code tái triển khai](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/baselines.py) |
| HLP-GCN-style reimplemented | Pose, causal | **0.335817 [C]** | **89.08±0.22** | **87.60±0.33** | **88.62±1.16** | **98.10±0.29** | **0.313±0.149** | — | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [code tái triển khai](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/baselines.py) |
| ST-GCN | Pose, causal | — | **89.45±0.65** | **87.94±0.72** | **82.64±0.43** | **98.06±0.41** | **0.328±0.060** | **85.53±0.39** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) |
| TCN | Pose, causal | — | **89.88±0.54** | **88.29±0.62** | **82.32±0.16** | **97.87±0.34** | **0.377±0.043** | **87.82±0.94** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) |
| GRU | Pose, causal | — | **88.82±0.43** | **87.10±0.45** | **81.99±0.82** | **98.14±0.31** | **0.345±0.058** | — | [RSC Table 1](https://arxiv.org/html/2608.02200v1) |
| RSC-Gesture-Pose (không VC) | Pose, causal | **0.335531 [C]** | **90.19±0.29** | **88.78±0.40** | **88.50±0.51** | **98.29±0.36** | **0.284±0.133** | **87.75±0.78** | [RSC Table 1](https://arxiv.org/html/2608.02200v1) · [model source](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) |

`[C]`: số tham số kiến trúc recognizer **đếm theo code tác giả** với `hidden=96`, 17 joints, 9 classes, `rgb_dim=12`. `*` Full RSC-GestureNet uses learned recognition backbone ~0.815M **plus calibration/selection components that have not been fully audited**; không ghi một số "exact full" giả. `†` 0.335531 là `RSCGestureBackbone` từ source, cần đối chiếu biến thể +VC với checkpoint. Tất cả phương pháp skeleton/RGB-pose chưa gồm pose-estimator nếu tính toàn bộ camera-to-command pipeline, nên **không xem 0.815M nhỏ hơn 6.96M theo cùng chuẩn full-pipeline**.

**VideoMamba continuous RAW có thêm:** Mean per-video FrameAcc **83.4973%**; Mean Temporal Overlap **0.653223**; Mean Gesture Recall **94.8035%**; batch forward ~**48.17 ms/window**, nhưng Offline Frame Accuracy **không cùng metric/giao thức** với window-level causal Accuracy của RSC. Nên để `—` cho Online-F1, TTC, Early@10 và Robust-F1 vì notebook chưa tính; **không suy từ Recall hoặc overlap**. RSC Table 1 báo **mean±standard deviation của ba seed**, VideoMamba đang chỉ có một checkpoint.