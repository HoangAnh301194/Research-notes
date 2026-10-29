# Báo cáo nghiên cứu ngày 22/10/2026

## A. Công việc đã làm
- Khảo sác các bài báo liên quan tới bộ IPN Hand và CTP Gesture 
- Bổ sugn thêm metric Accuracy@ 
- Bổ sung keyframe lỗi 

**Quy ước:** [O] nhóm tự đo trong notebook;
[P] tác giả công bố; 
[E] nhóm tính/đếm từ kiến trúc hoặc mã; 
[R] số từ model zoo/kiến trúc tham chiếu **không xác nhận đúng variant paper**; 
[D] phép cộng module đã biết; NR = chưa có hoặc chưa xác minh. **Params** tính bằng triệu tham số (M).

**Metric P1:** các notebook mới đã có code cho Accuracy@10/20/50/75/100%, Top-2/3, BalancedAcc, F1 theo lớp, F1@10/25/50, Edit, calibration và error visualization. 
**Hai bản notebook P1 trên GitHub hiện chưa lưu outputs**; chỉ các metric có trong log Kaggle đã cung cấp mới được điền số. Keyframe/filmstrip là **trực quan hóa 16 frame lấy mẫu**, không phải attention-derived keyframe.



## Bảng 1 — IPN-Hand: Isolated Gesture Recognition

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Kết quả và metric mới | Lưu ý giao thức |
|---|---|---|---|---|---|---|
| **VideoMamba-Tiny [O]** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB 16×224² | **6.9608** model | **29.0261 [E]** /clip | **Accuracy 84.65%** (932/1.101); **GT-prefix AUC 0.72399**; từng dự đoán đúng **88.56%**; Top-2/3, BalancedAcc, F1 theo lớp: **NR (P1)** | 1.101 GT clips; AUC trên observation ratio 10–100%, **không phải Early@10 online**; 169 mẫu sai; Accuracy@10/20/50 cần CSV chi tiết |
| CoboGesture + **VideoMAEv2** [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, ViT-L | ≈303.8 recognizer | 596.8327 [E] /clip | Accuracy **84.59%** | Gần tác vụ; sampling/preprocessing khác; GFLOPs nhóm tự ước lượng |
| CoboGesture + VideoMAE [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, ViT-L | ≈303.8 recognizer | 596.8327 [E] /clip | Accuracy **78.81%** | Cùng paper, backbone v1; GFLOPs tham khảo |
| Kausar et al. [Bài báo](https://doi.org/10.32604/cmc.2026.084514) | 2026 | RGB+Flow+Seg+geometry+Transformer | NR full; ResNet-34 **21.797672 [R]** | NR | Accuracy **92.34%** | Chưa khóa split; Params ResNet-34 chỉ tham chiếu, không gồm U-Net decoder/đầu nhận dạng |
| Gil-Martín et al. [Bài báo](https://doi.org/10.5220/0013053500003890) | 2025 | MediaPipe landmarks+LSTM | **0.058614** recognizer | NR | Accuracy **84.66±1.09%**; nhánh 37/13 subjects **86.38±2.03%** | 10-fold subject-wise CV hoặc split 37/13; không gồm MediaPipe; khác protocol |
| IPN original: ResNeXt-101 RGB+Flow [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+SPyNet, 32f | 47.56 recognizer; **49.000 [E]** pipeline | NR | Accuracy **86.32%** | Official IPN, khác modality/sampling |
| IPN original: ResNeXt-101 RGB+Seg [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+HarDNet, 32f | 47.56 recognizer; **51.674 [E]** pipeline | NR | Accuracy **84.77%** | Official IPN, khác modality/sampling |
| IPN original: ResNeXt-101 RGB [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB, 32f | 47.51 recognizer | NR | Accuracy **83.59%** | Cùng dữ liệu gốc, khác số frame |

## Bảng 2 — IPN-Hand: Continuous Gesture Recognition

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Kết quả và metric mới | Lưu ý giao thức |
|---|---|---|---|---|---|---|
| **VideoMamba Raw [O]** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB video, offline 64/16 | **6.9608** model | 29.0261 [E] /clip | **Mean FrameAcc 83.4889%**, Global FrameAcc **83.7308%**, Overlap **0.491110**, Recall **78.8995%**; F1@10/25/50, Edit, FP/min: **NR (P1)** | 52 test videos; raw; Levenshtein chưa có; segmental-F1 và Edit của notebook P1 chưa có log |
| VideoMamba R3 [O] [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB, post-processing 64/16 | 6.9608 | 29.0261 [E] /clip | Mean FrameAcc **83.5249%**, Overlap **0.507705**, Recall **76.2844%** | **Đã chọn ngưỡng trên test**, không dùng làm independent SOTA |
| CoboGesture + VideoMAEv2 [Bài báo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, offline 64/30 | ≈303.8 recognizer | 596.8327 [E] /clip | Mean FrameAcc **83.00%**, Overlap **0.510** | Khác stride; cần xác nhận công thức overlap tương ứng |
| CoboGesture + VideoMAE [Bài báo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, offline 64/30 | ≈303.8 recognizer | 596.8327 [E] /clip | Mean FrameAcc **80.00%**, Overlap **0.370** | Khác stride; không có Gesture Recall |
| Nayan et al. [Bài báo](https://doi.org/10.1007/s00138-024-01565-9) | 2024 | Multi-stream RGB+Flow+EHTD (+Seg), GoogLeNet+BiLSTM | NR full; GoogLeNet **6.624904 [R]** /backbone | NR | **Levenshtein 73.71%** (có Seg) / **69.40%** (không Seg) | Không phải FrameAcc; backbone khác toàn pipeline |
| TY-Net [Bài báo](https://doi.org/10.1109/ACCESS.2023.3341702) | 2023 | MediaPipe skeleton+Transformer | ≈5 [P] recognizer | 0.18 [P] recognizer | **Levenshtein 60.46%** | Online segmentation; Params/GFLOPs chưa tính MediaPipe |
| TMMF [Bài báo](https://doi.org/10.1109/TIP.2021.3108349) | 2021 | RGB+Flow, fusion theo thời gian | NR | NR | **Levenshtein 68.12%** | Số từ [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9); khác metric FrameAcc |
| TD-Net [Bài báo](https://doi.org/10.1109/ICCAIS56082.2022.9990122) | 2022 | RGB→MediaPipe skeleton + detector/classifier | NR | NR | **Levenshtein 40.10%** | Dùng skeleton; không phải FrameAcc |
| IPN original: ResNeXt+ResLight+SPyNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+Flow, original continuous | 47.56 recognizer; **49.908 [E]** pipeline | NR | **Levenshtein 42.47%** | Original IPN Table IX; không phải FrameAcc |
| IPN original: ResNeXt+ResLight+HarDNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+Seg, original continuous | 47.56 recognizer; **52.582 [E]** pipeline | NR | **Levenshtein 39.01%** | Original Table IX; khác modality |
| IPN original: ResNeXt+ResLight [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB, original continuous | 47.51 recognizer; **48.405 [E]** pipeline | NR | **Levenshtein 25.34%** | Original Table IX; detector ResLight |

## Bảng 3 — CTPGesture v1: Isolated Recognition và đối chứng ảnh dẫn xuất

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Kết quả và metric mới | Lưu ý giao thức |
|---|---|---|---|---|---|---|
| **VideoMamba-Tiny [O]** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB video, 16×224² | **6.959817** model | **29.0261 [E]** /clip | **Accuracy 98.5532%**, **Macro-F1 97.9590%**, BalancedAcc **98.1226%**; gesture-only Acc **98.0210%**, Macro-F1 **98.2363%**; Top-2/3, Accuracy@10/20/50: **NR (P1)** | **11 video / 1.728 GT segments**, 869 no-gesture, 859 gesture; full 9-class vs gesture-only 8-class khác phạm vi |
| **Zhang — BiLSTM–Transformer** [Bài báo](https://doi.org/10.3390/electronics15122578) | 2026 | **Skeleton:** YOLO11m-Pose→40-frame BiLSTM–Transformer | **0.08641 [P]** head; ≈20.98641 [D/R] kèm YOLO-Pose | NR total; YOLO-Pose 71.5 [R] /frame @640 | **Accuracy 98.91%** | 10 video test; sequence classification; không cùng 11-video split; head ≠ pipeline |
| **FDMB-YOLOv11** [Bài báo](https://doi.org/10.1371/journal.pone.0357212) | 2026 | **Non-skeleton:** RGB frame, YOLO11n/MobileNetV4/BiFormer | **1.93 [P]** model | NR | **mAP@0.5 ≈97.17%** | Ảnh trích từ nguồn CTP liên quan, chia lại tập; **object detection mAP ≠ isolated Accuracy** |
| **Liu — Improved YOLOv8n** [Bài báo](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) | 2026 | **Non-skeleton:** RGB frame, YOLOv8n + attention | **1.8 [P]** model | **5.7 [P]** | **mAP 97.9%** | Ảnh trích từ CTP + ảnh Internet, dataset 16.000 ảnh; **không cùng 1.728 segments** |
| **Assiri & Selim — Swin + CNN–BiLSTM** [Bài báo](https://doi.org/10.1016/j.asej.2025.103383) | 2025 | **Non-skeleton ứng viên:** RGB, Swin Transformer + CNN–BiLSTM | NR | NR | **Accuracy 99.25%** | **Chưa xác minh đúng CTPGesture v1**; chỉ biết tên *Traffic Police Gesture Dataset*; không xếp hạng |
| **P3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, Pseudo-3D CNN | ≈65.3 [R] **P3D-199** | NR | **Accuracy 92.21%** | 9 test videos / 1.565 clips; **variant P3D chưa khóa**; baseline trong bài Xiong |
| **C3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, C3D | 78.4 [R] MMAction2 | 38.5 [R] /16×112² | **Accuracy 87.12%** | 9 test videos / 1.565 clips; Params/GFLOPs model zoo không xác nhận checkpoint Xiong |
| **3D ResNet** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, 3D ResNet | 33.371472 [R] **R3D-18** | 40.70 [R] /16×112² | **Accuracy 81.16%** | 9 test videos / 1.565 clips; **không biết depth** của bản Xiong, số tham chiếu R3D-18 |

## Bảng 4 — CTPGesture v1: Continuous Gesture Recognition

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Kết quả và metric mới | Lưu ý giao thức |
|---|---|---|---|---|---|---|
| **VideoMamba Raw [O]** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB video, **offline** 32/8 | **6.959817** model | 29.0261 [E] /clip | **Global FrameAcc 83.9836%**, **Macro-F1 80.6135%**, Mean FrameAcc 83.4973%, Overlap **0.653223**, Recall **94.8035%**; **F1@10/25/50, Edit, ECE, TP/FP/FN: NR (P1)** | 11 videos / 67.356 frames; offline overlapping windows có thể dùng future frame; chưa có Online-F1/TTC |
| VideoMamba R3 [O] [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB offline 32/8 + post-processing | 6.959817 | 29.0261 [E] /clip | Global FrameAcc **83.9940%**, Macro-F1 **80.6254%**, Overlap **0.6590**, Recall **94.45%** | **Tuned trên official test**, không phải held-out; không dùng để khẳng định SOTA |
| **RSC-GestureNet** [Bài báo](https://arxiv.org/abs/2608.02200) | 2026 | **Skeleton + RGB**, causal 32/4, 3 seeds | ≥0.815270 [E] recognizer fusion; thiếu pose extractor | NR | **Accuracy 93.33±0.24%**, **Macro-F1 91.71±0.27%**, Online-F1 **91.69±0.29%**, Early@10 **98.80±0.07%**, TTC **0.153±0.013s** | Causal, selective/calibrated, đo theo output steps; **không cùng offline FrameAcc**, Early@10 ≠ GT-prefix Accuracy@10% |

## Danh sách bài báo khảo sát
Các điều kiện để chọn danh sách : 
  - Là các công trình trong những năm gần đây hoặc bài báo công bố dataset 
  - Có một vài công trình sử dụng skeleton 
  - Đa số là none skeleton

**IPN-Hand — 8 bài báo/công trình:** **(1)** [CoboGesture 2026](https://doi.org/10.1109/ACCESS.2026.3664027), **(2)** [Kausar et al. 2026](https://doi.org/10.32604/cmc.2026.084514), **(3)** [Gil-Martín et al. 2025](https://doi.org/10.5220/0013053500003890), **(4)** [IPN-Hand original 2020/21](https://doi.org/10.1109/ICPR48806.2021.9412317), **(5)** [TY-Net 2023](https://doi.org/10.1109/ACCESS.2023.3341702), **(6)** [Nayan et al. 2024](https://doi.org/10.1007/s00138-024-01565-9), **(7)** [TMMF 2021](https://doi.org/10.1109/TIP.2021.3108349), **(8)** [TD-Net 2022](https://doi.org/10.1109/ICCAIS56082.2022.9990122). Một bài có thể có nhiều biến thể hoặc xuất hiện ở cả isolated và continuous.

**CTPGesture v1 / nguồn ảnh liên quan — 8 phương pháp khảo sát sâu:** **2 skeleton**: **(1)** [Zhang 2026 — YOLO11m-Pose + BiLSTM–Transformer](https://doi.org/10.3390/electronics15122578), **(2)** [RSC-GestureNet 2026](https://arxiv.org/abs/2608.02200). **6 non-skeleton**: **(3)** [FDMB-YOLOv11 2026](https://doi.org/10.1371/journal.pone.0357212), **(4)** [Liu 2026 — YOLOv8n](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007), **(5)** [Assiri & Selim 2025 — Swin + CNN–BiLSTM](https://doi.org/10.1016/j.asej.2025.103383) *(ứng viên, chưa xác minh đúng v1)*, **(6)** P3D, **(7)** C3D, **(8)** 3D ResNet. Ba baseline cuối đều từ **[Xiong 2021, Table 3](https://doi.org/10.3390/electronics10050551)**, nên **8 phương pháp không đồng nghĩa 8 bài báo độc lập**.


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
