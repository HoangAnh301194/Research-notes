# Báo cáo nghiên cứu ngày 22/10/2026

## A. Công việc đã làm
- Khảo sác các bài báo liên quan tới bộ IPN Hand và CTP Gesture 
- Bổ sugn thêm metric Accuracy@ 
- Bổ sung keyframe lỗi 



## Danh sách bài báo và công trình đã khảo sát

Danh mục được trình bày **trước các bảng so sánh**, bao gồm nghiên cứu gần đây, công trình công bố bộ dữ liệu và phương pháp đối chứng. Phần lớn sử dụng dữ liệu ảnh/video (non-skeleton), đồng thời giữ một số phương pháp dùng đặc trưng khung xương (skeleton) để đánh giá sự khác biệt về kiến trúc và giao thức.

### 1. IPN-Hand — 8 bài báo/công trình

1. [**CoboGesture: A Continuous Hand Gesture Dataset and Recognition Method for Human-Collaborative Robot Interaction**](https://doi.org/10.1109/ACCESS.2026.3664027) (2026) — Đánh giá các biến thể VideoMAE/VideoMAEv2; có số liệu trên IPN-Hand.
2. [**Deep Hand Segmentation and Multi-Modal Gesture Recognition for Human-Robot Interaction via 3D Volumetric Encoding**](https://doi.org/10.32604/cmc.2026.084514) (2026) — Kết hợp RGB, optical flow, phân đoạn bàn tay và đặc trưng hình học.
3. [**Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks**](https://doi.org/10.5220/0013053500003890) (2025) — Dùng MediaPipe hand landmarks và LSTM; khảo sát các giao thức chia dữ liệu.
4. [**IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition**](https://doi.org/10.1109/ICPR48806.2021.9412317) (2021; hội nghị ICPR 2020) — Bài công bố bộ dữ liệu và các baseline RGB, optical flow, segmentation.
5. [**TY-Net: Transforming YOLO for Hand Gesture Recognition**](https://doi.org/10.1109/ACCESS.2023.3341702) (2023) — Phương pháp dùng skeleton và mạng nhận dạng gọn nhẹ.
6. [**A multi-modal framework for continuous and isolated hand gesture recognition utilizing movement epenthesis detection**](https://doi.org/10.1007/s00138-024-01565-9) (2024) — Nhận dạng isolated và continuous với nhiều luồng đặc trưng, loại khung chuyển tiếp.
7. [**TMMF: Temporal Multi-Modal Fusion for Single-Stage Continuous Gesture Recognition**](https://doi.org/10.1109/TIP.2021.3108349) (2021) — Kết hợp nhiều modality theo thời gian trong nhận dạng liên tục.
8. [**A Continuous Real-time Hand Gesture Recognition Method based on Skeleton**](https://doi.org/10.1109/ICCAIS56082.2022.9990122) (2022) — Nhận dạng cử chỉ liên tục dựa trên skeleton và TD-Net.

**Lưu ý:** Một bài báo có thể có nhiều biến thể và đánh giá ở cả isolated lẫn continuous. Các biến thể được thể hiện riêng trong bảng nhưng không được đếm thành bài báo độc lập.

### 2. CTPGesture v1 và các nghiên cứu ảnh dẫn xuất — 6 bài báo/công trình độc lập

1. [**A Traffic Police Gesture Recognition Method Based on BiLSTM-Transformer Architecture**](https://doi.org/10.3390/electronics15122578) (2026) — Sử dụng keypoint từ YOLO11m-Pose cùng BiLSTM–Transformer.
2. [**RSC-GestureNet: Reliability-Aware Selective Causal Recognition of Chinese Traffic Police Gestures**](https://arxiv.org/abs/2608.02200) (2026) — Kết hợp skeleton/RGB và độ tin cậy pose theo giao thức causal.
3. [**FDMB-YOLOv11: Traffic police command gesture recognition method under different lighting conditions**](https://doi.org/10.1371/journal.pone.0357212) (2026) — Phát hiện trên ảnh trong các điều kiện chiếu sáng; không phải phân loại video.
4. [**Lightweight traffic police gesture recognition method for autonomous driving**](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) (2026) — Phương pháp YOLOv8n nhẹ cho nhận dạng/phát hiện trên ảnh.
5. [**A swin transformer-driven framework for gesture recognition to assist hearing impaired people by integrating deep learning with secretary bird optimization algorithm**](https://doi.org/10.1016/j.asej.2025.103383) (2025) — Nghiên cứu liên quan về cử chỉ ngôn ngữ ký hiệu; **chưa xác minh dùng CTPGesture v1**, không coi là baseline cùng dataset.
6. [**Traffic Police Gesture Recognition Based on Gesture Skeleton Extractor and Multichannel Dilated Graph Convolution Network**](https://doi.org/10.3390/electronics10050551) (2021) — Công trình có phương pháp chính dựa trên skeleton; bảng đối chứng còn chứa **P3D, C3D và 3D ResNet**, là ba baseline xử lý RGB video.

**Lưu ý về phạm vi:** Danh mục này gồm nghiên cứu video, nghiên cứu detection ảnh liên quan và một bài về cử chỉ ngôn ngữ ký hiệu (Assiri & Selim) cần xác minh tính tương thích dữ liệu. Bài Xiong (2021) đề xuất phương pháp dựa trên skeleton nhưng cung cấp ba baseline xử lý RGB (P3D, C3D, 3D ResNet). Vì thế **8 phương pháp đối chứng không đồng nghĩa 8 bài báo độc lập**, và không được so sánh trực tiếp mAP của detection ảnh với Accuracy của nhận dạng video.

## Kết quả so sánh chính

> **Nguyên tắc đọc:** Các phương pháp trong cùng bảng không nhất thiết cùng tập test, chia dữ liệu, chế độ online/offline hoặc thước đo. Các giá trị là **kết quả báo cáo theo từng giao thức**, không phải bảng xếp hạng công bằng. Dấu `-` nghĩa là chưa có số liệu đúng định nghĩa đã xác minh. Params/GFLOPs có thể chỉ tính một thành phần, xem phụ lục để biết giới hạn.

### Bảng 1 — IPN-Hand: Nhận dạng đoạn cử chỉ (isolated)

| Phương pháp / nguồn | Đầu vào | Params (M) | GFLOPs | N mẫu | Accuracy (%) | GT-prefix AUC | Ghi chú ngắn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB 16×224² | 6.9608 | 29.0261 | 1.101 | 84.65 | 0.72399 | VideoMamba; RGB 16 frames; GT-prefix offline. Chi phí theo clip, không phải toàn hệ thống. |
| CoboGesture + **VideoMAEv2** [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, ViT-L | ≈303.8 (recognizer) | 596.8327 | - | 84.59 | - | Khác cách lấy mẫu và tiền xử lý so với VideoMamba. |
| CoboGesture + VideoMAE [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, ViT-L | ≈303.8 (recognizer) | 596.8327 | - | 78.81 | - | Biến thể VideoMAE đời trước, cùng bài CoboGesture. |
| Kausar et al. [Nguồn](https://doi.org/10.32604/cmc.2026.084514) | RGB + Flow + Seg + Transformer | ≈21.284672 (encoder tham chiếu) | ResNet-34 3.66/ảnh (tham chiếu) | - | 92.34 | - | Chưa khóa split; tham số chỉ phản ánh ResNet-34 encoder. |
| Gil-Martín et al. [Nguồn](https://doi.org/10.5220/0013053500003890) | MediaPipe landmarks+LSTM | 0.058614 (LSTM) | Palm192 0.354 /call; hand landmark + LSTM: - | - | 84.66±1.09; 86.38±2.03 | - | **Hai giao thức:** 10-fold CV / 37 train–13 test; 0,058614M chỉ LSTM. |
| IPN original: ResNeXt-101 RGB+Flow [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+SPyNet, 32f | 47.56 + 1.440 | - | - | 86.32 | - | RGB + optical flow; khác dữ liệu đầu vào và số khung. |
| IPN original: ResNeXt-101 RGB+Seg [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+HarDNet, 32f | 47.56 + 4.114 | - | - | 84.77 | - | RGB + segmentation; khác dữ liệu đầu vào và số khung. |
| IPN original: ResNeXt-101 RGB [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB, 32f | 47.51 | - | - | 83.59 | - | Baseline RGB; 32 frames, khác thiết lập 16 frames. |

**Lưu ý:** Gil-Martín có **hai giao thức độc lập** (10-fold CV và 37 train/13 test); hai Accuracy trong cùng ô theo đúng thứ tự này. Những số Params chỉ là *recognizer* hoặc backbone được ghi rõ, không nên so trực tiếp với chi phí toàn pipeline.

### Bảng 2 — IPN-Hand: Nhận dạng liên tục (continuous)

| Phương pháp / nguồn | Đầu vào / giao thức | Params (M) | Accuracy bài báo (%) | FrameAcc TB/video (%) | Overlap | Gesture Recall (%) | Ghi chú ngắn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB video, offline 64/16 | 6.9608 | - | 83.4889 | 0.491110 | 78.8995 | RAW offline 64/16; không phải đánh giá online. |
| VideoMamba R3 [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB, post-processing 64/16 | 6.9608 | - | 83.5249 | 0.507705 | 76.2844 | R3 chọn ngưỡng trên test; không dùng để khẳng định hơn baseline. |
| CoboGesture + VideoMAEv2 [Nguồn](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, offline 64/30 | ≈303.8 (recognizer) | - | 83.00 | 0.510 | - | Offline 64/30; cách tính overlap cần đối chiếu. |
| CoboGesture + VideoMAE [Nguồn](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, offline 64/30 | ≈303.8 (recognizer) | - | 80.00 | 0.370 | - | Offline 64/30; cách tính overlap cần đối chiếu. |
| Nayan et al. [Nguồn](https://doi.org/10.1007/s00138-024-01565-9) | RGB + Flow + EHTD + Seg | ≈6.624904 (backbone tham chiếu) | 73.71 | - | - | - | Giá trị bài báo là **Levenshtein Accuracy**, không phải FrameAcc. |
| Nayan et al. [Nguồn](https://doi.org/10.1007/s00138-024-01565-9) (không Seg) | RGB + Flow + EHTD | ≈6.624904 (backbone tham chiếu) | 69.40 | - | - | - | Không dùng segmentation; giá trị là **Levenshtein Accuracy**. |
| TY-Net [Nguồn](https://doi.org/10.1109/ACCESS.2023.3341702) | Skeleton + Transformer | ≈5 (recognizer) | 60.46 | - | - | - | Skeleton; kết quả online theo Levenshtein, khác FrameAcc. |
| TMMF [Nguồn](https://doi.org/10.1109/TIP.2021.3108349) | RGB+Flow, fusion theo thời gian | - | 68.12 | - | - | - | Levenshtein Accuracy trích từ bảng của Nayan. |
| TD-Net [Nguồn](https://doi.org/10.1109/ICCAIS56082.2022.9990122) | Skeleton + TD-Net | Chưa xác minh full TD-Net | 40.10 | - | - | - | Skeleton; kết quả theo Levenshtein, khác FrameAcc. |
| IPN original: ResNeXt+ResLight+SPyNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+Flow, original continuous | 47.56 + 0.908 + 1.440 | 42.47 | - | - | - | Benchmark gốc; Levenshtein Accuracy, không phải FrameAcc. |
| IPN original: ResNeXt+ResLight+HarDNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+Seg, original continuous | 47.56 + 0.908 + 4.114 | 39.01 | - | - | - | Benchmark gốc với segmentation; Levenshtein Accuracy. |
| IPN original: ResNeXt+ResLight [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB, original continuous | 47.51 + 0.895 | 25.34 | - | - | - | Benchmark gốc chỉ RGB; Levenshtein Accuracy. |

**Lưu ý:** Cột Accuracy bài báo của Nayan, TY-Net, TMMF, TD-Net và IPN original chủ yếu là **Levenshtein Accuracy**, **không tương đương FrameAcc**. VideoMamba và CoboGesture có thể dùng cách tính overlap khác nhau.

### Bảng 3 — CTPGesture v1: Nhận dạng đoạn cử chỉ và đối chứng liên quan

| Phương pháp / nguồn | Đầu vào | Params (M) | N mẫu | Accuracy (%) | Loại bài toán | Ghi chú ngắn |
| --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB video, 16×224² | 6.959817 | 1.728 | 98.5532 | Phân loại video | VideoMamba 9 lớp, 11 test videos; 1.728 GT clips. |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) — Gesture only (GT 1–8) | RGB video, 16×224² | 6.959817 | 859 | 98.0210 | Phân loại video | Subset 859 gesture GT từ cùng mô hình 9 lớp. |
| **Zhang — BiLSTM–Transformer** [Nguồn](https://doi.org/10.3390/electronics15122578) | Skeleton; 40 frames | 20.9 (pose) + 0.08641 (head) | - | 98.91 | Video skeleton | Skeleton; 10 test videos; split khác VideoMamba. |
| **FDMB-YOLOv11** [Nguồn](https://doi.org/10.1371/journal.pone.0357212) | RGB frame; YOLO11n | 1.93 | - | - | Detection ảnh | **Detection ảnh**; mAP không thể điền vào Accuracy video. |
| **Liu — Improved YOLOv8n** [Nguồn](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) | RGB frame; YOLOv8n | 1.8 | - | - | Detection ảnh | **Detection ảnh**; dùng tập ảnh xử lý lại. |
| **Assiri & Selim — Swin + CNN–BiLSTM** [Nguồn](https://doi.org/10.1016/j.asej.2025.103383) | RGB; Swin + CNN–BiLSTM | ≈28.288354 (Swin-T tham chiếu) | - | 99.25 | Video; chưa khóa dataset | Chưa xác minh sử dụng đúng CTPGesture v1. |
| **P3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, Pseudo-3D CNN | ≈65.3 (P3D-199 tham chiếu) | 1.565 | 92.21 | Phân loại video | Baseline trong Xiong, 9 video / 1.565 clips; variant tham chiếu. |
| **C3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, C3D | 78.4 (tham chiếu) | 1.565 | 87.12 | Phân loại video | Baseline Xiong; cấu hình model zoo chỉ tham khảo. |
| **3D ResNet** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, 3D ResNet | 33.371472 (R3D-18 tham chiếu) | 1.565 | 81.16 | Phân loại video | Baseline Xiong; độ sâu mạng gốc chưa xác minh. |

**Lưu ý:** Các công trình **detection trên ảnh** (FDMB-YOLOv11, Liu) không có Accuracy phân loại video tương ứng. Bản Assiri & Selim chưa xác minh dùng đúng CTPGesture v1; các baseline Xiong có split khác.

### Bảng 4 — CTPGesture v1: Nhận dạng liên tục

| Phương pháp / nguồn | Đầu vào / giao thức | Params (M) | Accuracy bài báo (%) | FrameAcc global (%) | FrameAcc TB/video (%) | Overlap | Ghi chú ngắn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB video, offline 32/8 | 6.959817 | - | 83.9836 | 83.4973 | 0.653223 | RAW offline 32/8; có sử dụng frame tương lai. |
| VideoMamba R3 [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB offline 32/8 + post-processing | 6.959817 | - | 83.9940 | - | 0.6590 | R3 tinh chỉnh trên official test; không phải held-out. |
| **RSC-GestureNet** [Nguồn](https://arxiv.org/abs/2608.02200) | Skeleton + RGB; causal 32/4 | 0.815270 (fusion only) | **93.33±0.24** | - | - | - | Causal 32/4, 3 seeds; **không so trực tiếp** với FrameAcc offline. |

**Lưu ý:** RSC-GestureNet đánh giá **causal**; VideoMamba là **offline** với cửa sổ chồng lấn. Không dùng Accuracy causal của RSC làm FrameAcc của VideoMamba.


## Phân tích chuyên sâu VideoMamba

Các bảng **5.1–5.5 chỉ trình bày số liệu thực nghiệm VideoMamba** mà nhóm đã có trên IPN-Hand và CTPGesture v1. Những bài báo đối chứng không có các metric P1 tương ứng **không được đưa vào phần này**; chúng đã xuất hiện trong Bảng 1–4. Không trộn kết quả *isolated* với *continuous*, không lấy metric RAW thay cho R3.

**Nguồn thực nghiệm đã chạy và có output (kiểm tra 02/11/2026):** [IPN-Hand — notebook P1](../261028/videomamba-ipn-hand-p1-metrics-keyframes-python313%20%281%29.ipynb) (16/18 ô code có output; 47 output), [CTPGesture — notebook P1](../261028/videomamba-ctpgesture-p1-metrics-keyframes-python3%20%283%29.ipynb) (19/19 ô; 51 output). Notebook đọc và trình bày kết quả: [IPN-Hand Complete Results](../261028/ipn-hand-complete-videomamba-p1-results-16-fra.ipynb) và [CTPGesture Complete Results](../261028/ctpgesture-v1-complete-videomamba-p1-results-1.ipynb). Các bản mã nguồn P1 trong `KaggleNotebook/` chưa lưu output, vì vậy **dẫn số thực nghiệm bằng notebook đã chạy trong `261028/`**. Hai notebook Complete Results không phải lần thử nghiệm độc lập.

### 5.1. Isolated — Accuracy theo phần cử chỉ đã quan sát

| Mô hình / tập test | N clip | Acc@10% | Acc@20% | Acc@50% | Acc@75% | Acc@100% | GT-prefix AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba-Tiny | 1.101 | 38.6921 | 53.7693 | 76.1126 | 83.8329 | 84.6503 | 0.72399 |
| CTP · VideoMamba (9 lớp) | 1.728 | 86.0532 | 93.1713 | 97.5694 | 98.3796 | 98.5532 | 0.963140 |
| CTP · VideoMamba (gesture-only) | 859 | 91.0361 | 95.4598 | 97.6717 | 97.9045 | 98.0210 | 0.969247 |

Đây là **GT-prefix offline**: đoạn cử chỉ được cắt theo nhãn đúng trước khi lấy phần đầu. Không đồng nhất với việc robot phát hiện lệnh trong thời gian thực (*causal Early@10*).

### 5.2. Isolated — F1, Top-K và mức độ nhận dạng sớm

| Mô hình / tập test | Macro-F1 (%) | Balanced Acc (%) | Top-2 (%) | Top-3 (%) | Ever-correct (%) | First-correct ratio | First-stable ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba-Tiny | 78.34 | 77.2581 | 95.0045 | 97.2752 | 88.5559 | 0.300308 | 0.315880 |
| CTP · VideoMamba (9 lớp) | 97.9590 | 98.1226 | 99.7685 | 99.8843 | 99.1319 | 0.135231 | 0.135232 |
| CTP · VideoMamba (gesture-only) | 98.2363 | 98.0030 | - | - | - | - | - |

Chỉ số *first-correct* và *first-stable* là tỷ lệ chiều dài đoạn theo giao thức GT-prefix; không phải độ trễ tính bằng giây.

### 5.3. Isolated — Calibration và minh chứng keyframe

| Mô hình / tập test | ECE-15 | Brier | NLL | Keyframe đã xuất |
| --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba-Tiny | 0.029848 | 0.197447 | 0.420977 | 1.101 JPEG: 932 đúng / 169 sai |
| CTP · VideoMamba (9 lớp) | 0.005987 | 0.021505 | 0.051804 | 1,728/1,728 JPEG; 1,703 đúng; 25 sai; 0 lỗi |
| CTP · VideoMamba (gesture-only) | - | - | - | Dùng chung 1,728 JPEG từ bộ 9 lớp; subset 859 gesture |

ECE, Brier và NLL chỉ đánh giá **calibration ở isolated**. Số keyframe là tổng trường hợp có log xuất hình đúng/sai, **không phải số video độc lập**. Xem ví dụ trong [báo cáo 28/10](../261028/Readme.md).

### 5.4. Continuous — FrameAcc, Macro-F1 và chất lượng phân đoạn

| Mô hình / tập test | FrameAcc global (%) | FrameAcc TB/video (%) | Macro-F1 (%) | Seg F1@10 (%) | Seg F1@25 (%) | Seg F1@50 (%) | Edit (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba RAW | 83.7308 | 83.4889 | 64.4012 | 79.4544 | 77.5789 | 69.9062 | 74.3831 |
| IPN-Hand · VideoMamba R3 | - | 83.5249 | - | - | - | - | - |
| CTP · VideoMamba RAW | 83.9836 | 83.4973 | 80.6135 | 96.0472 | 94.7493 | 85.5457 | 93.4968 |
| CTP · VideoMamba R3 | 83.9940 | - | 80.6254 | - | - | - | - |

Với R3, các metric phân đoạn chưa được xác minh thì để `-`, **không chuyển kết quả RAW sang R3**. FrameAcc theo video khác cách tổng hợp FrameAcc trên tất cả frame.

### 5.5. Continuous — Ghép đoạn, sai lệch biên và keyframe

| Mô hình / tập test | Precision@50 (%) | Recall@50 (%) | Onset MAE (s) | Offset MAE (s) | FP/min | Temporal overlap | Keyframe đã xuất |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba RAW | 65.8635 | 74.4777 | 0.440922 | 0.481819 | 3.596927 | 0.491110 | 1.526 JPEG: 820 ghép đúng / 706 lỗi |
| IPN-Hand · VideoMamba R3 | - | - | - | - | - | 0.507705 | 16f + GT/Pred timeline: code |
| CTP · VideoMamba RAW | 86.7225 | 84.4005 | 0.290115 | 0.445517 | 1.483164 | 0.653223 | 970/970 JPEG; 725 matched; 245 lỗi; 0 lỗi xuất |
| CTP · VideoMamba R3 | - | - | - | - | - | 0.658979 | 16f + GT/Pred timeline: code |

**FP/min** trong các notebook P1 RAW được tính từ đoạn dự đoán không ghép được mỗi phút (ngưỡng tIoU 0,50), không mặc định trùng định nghĩa FP/min của mọi bài báo. Các sai số onset/offset MAE tính trên những đoạn đã ghép đúng.


## Phụ lục — Bảng đầy đủ, nguồn và giả định tính toán

Để báo cáo chính gọn và dễ theo dõi, **4 bảng đối chứng gốc** với thông số đầy đủ, chú thích về giao thức và nguồn kiểm chứng được lưu trong các phụ lục dưới đây. **Bảng P1 40 cột đã được thay bằng các bảng 5.1–5.5 chỉ chứa kết quả thực nghiệm VideoMamba**, không lặp lại những công trình không có metric P1 tương ứng.

<details>
<summary>Phụ lục 1 — IPN-Hand: Isolated Gesture Recognition (bảng và chú thích đầy đủ)</summary>

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Params full pipeline (M) | GFLOPs full pipeline | N mẫu | Accuracy (%) | Accuracy@100% (%) | GT-prefix AUC | Ever-correct (%) | Chú thích / nguồn số liệu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB 16×224² | **6.9608** model | **29.0261** /clip | 6.9608 | 29.0261 /clip | 1.101 | 84.65 | 84.6503 | 0.72399 | 88.56 | 1.101 GT clips; GT-prefix AUC dùng 10–100%, không phải Early@10 causal; 169 mẫu sai; Accuracy@10/20/50/75 đã được đo trong notebook P1 261028; BalancedAcc = trung bình recall 13 lớp từ bảng gt_idx/ samples/correct trong [notebook Cell 12](../261007/videomamba_man_configsBenchmark.ipynb); **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| CoboGesture + **VideoMAEv2** [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, ViT-L | ≈303.8 recognizer | 596.8327 /clip | 303.8 RGB recognizer | 596.8327 /clip | - | 84.59 | - | - | - | Gần tác vụ; sampling/preprocessing khác; GFLOPs nhóm tự ước lượng; **Nguồn số liệu:** Params full pipeline (M): do tác giả công bố; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| CoboGesture + VideoMAE [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, ViT-L | ≈303.8 recognizer | 596.8327 /clip | 303.8 RGB recognizer | 596.8327 /clip | - | 78.81 | - | - | - | Cùng paper, backbone v1; GFLOPs tham khảo; **Nguồn số liệu:** Params full pipeline (M): do tác giả công bố; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| Kausar et al. [Bài báo](https://doi.org/10.32604/cmc.2026.084514) | 2026 | RGB+Flow+Seg+geometry+Transformer | ResNet-34 encoder ≈21.284672 | ResNet-34 3.66 /224² (classifier) | - | - | - | 92.34 | - | - | - | Chưa khóa split; Params ResNet-34 chỉ tham chiếu, không gồm U-Net decoder/đầu nhận dạng; ResNet34 encoder ≈21.797672 − 0.513000M FC; U-Net decoder, Ef-RAFT, Transformer chưa có cấu hình để cộng full; [nguồn](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html); **Nguồn số liệu:** GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo; Params (M): nhóm ước lượng từ các module tham chiếu, chưa xác minh toàn pipeline. |
| Gil-Martín et al. [Bài báo](https://doi.org/10.5220/0013053500003890) | 2025 | MediaPipe landmarks+LSTM | LSTM 0.058614 ; Hands landmark Full 1.98 / Lite 1.00 ; Palm192 1.133 | Palm192 0.354 /call; hand landmark + LSTM: - | ≈3.171614 Full; ≈2.191614 Lite | - | - | 84.66±1.09 (10-fold CV); 86.38±2.03 (37/13 subjects) | - | - | - | **Hai giao thức đánh giá độc lập:** 10-fold subject-wise CV: **84.66±1.09%**; chia 37 subjects huấn luyện / 13 subjects kiểm thử: **86.38±2.03%**. Đây là cùng kiến trúc, khác phương án chia dữ liệu; không coi là hai phương pháp và không gộp hai Accuracy thành một chỉ số. recognizer không gồm MediaPipe; khác split và giao thức; 0.058614M chỉ LSTM recognizer; MediaPipe chưa cộng.; Giá trị full là **ước lượng theo 2 biến thể giả định**, KHÔNG phải cấu hình paper xác nhận: 0.058614+1.133+1.98=3.171614M (Full) hoặc +1.00=2.191614M (Lite). Bài xác nhận dùng MediaPipe 21 hand landmarks nhưng không nêu checkpoint/độ phức tạp. Palm192 từ [Luxonis](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3), Hands 1.00/1.98 từ [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/); [MediaPipe Hands](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md) gồm cả detector+landmark; **Nguồn số liệu:** Params (M): notebook và phép tính từ mã; Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo; Params full pipeline (M): nhóm ước lượng từ các module tham chiếu, chưa xác minh toàn pipeline. |
| IPN original: ResNeXt-101 RGB+Flow [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+SPyNet, 32f | ResNeXt 47.56 + SPyNet 1.440 | - | 49.000 = 47.56 + 1.440 | - | - | 86.32 | - | - | - | Official IPN, khác modality/sampling; IPN original Tables IV–V công bố Params của các module riêng; total là phép cộng; **Nguồn số liệu:** Params (M): do tác giả công bố; Params full pipeline (M): nhóm cộng thông số các module được công bố. |
| IPN original: ResNeXt-101 RGB+Seg [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+HarDNet, 32f | ResNeXt 47.56 + HarDNet 4.114 | - | 51.674 = 47.56 + 4.114 | - | - | 84.77 | - | - | - | Official IPN, khác modality/sampling; IPN original Tables IV–V công bố Params của các module riêng; total là phép cộng; **Nguồn số liệu:** Params (M): do tác giả công bố; Params full pipeline (M): nhóm cộng thông số các module được công bố. |
| IPN original: ResNeXt-101 RGB [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB, 32f | 47.51 recognizer | - | 47.51 | - | - | 83.59 | - | - | - | Cùng dữ liệu gốc, khác số frame; **Nguồn số liệu:** Params full pipeline (M): do tác giả công bố. |

</details>

<details>
<summary>Phụ lục 2 — IPN-Hand: Continuous Gesture Recognition (bảng và chú thích đầy đủ)</summary>

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Params full pipeline (M) | GFLOPs full pipeline | N video | Accuracy (paper, %) | FrameAcc global (%) | FrameAcc TB/video (%) | Temporal Overlap | Gesture Recall (%) | FP/min | Chú thích / nguồn số liệu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB video, offline 64/16 | **6.9608** model | 29.0261 /clip | 6.9608 | 29.0261 /clip | 52 | - | 83.7308 | 83.4889 | 0.491110 | 78.8995 | - | Raw; offline 64/16; 52 test videos. Levenshtein diagnostic, F1@10/25/50, Edit và FP/min RAW đã có output P1 261028; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| VideoMamba R3 [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | 2026 | RGB, post-processing 64/16 | 6.9608 | 29.0261 /clip | 6.9608 | 29.0261 /clip | 52 | - | - | 83.5249 | 0.507705 | 76.2844 | - | **Đã chọn ngưỡng trên test**, không dùng làm independent SOTA; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| CoboGesture + VideoMAEv2 [Bài báo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, offline 64/30 | ≈303.8 recognizer | 596.8327 /clip | 303.8 RGB recognizer | 596.8327 /clip | - | - | - | 83.00 | 0.510 | - | - | Khác stride; cần xác nhận công thức overlap tương ứng; **Nguồn số liệu:** Params full pipeline (M): do tác giả công bố; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| CoboGesture + VideoMAE [Bài báo](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | RGB video, offline 64/30 | ≈303.8 recognizer | 596.8327 /clip | 303.8 RGB recognizer | 596.8327 /clip | - | - | - | 80.00 | 0.370 | - | - | Khác stride; không có Gesture Recall; **Nguồn số liệu:** Params full pipeline (M): do tác giả công bố; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| Nayan et al. [Bài báo](https://doi.org/10.1007/s00138-024-01565-9) | 2024 | Multi-stream RGB+Flow+EHTD (+Seg), GoogLeNet+BiLSTM | GoogLeNet ≈6.624904 /backbone | GoogLeNet 1.50 /224² /backbone | - | - | - | 73.71 | - | - | - | - | - | Có Seg; metric Levenshtein, không phải FrameAcc; GoogLeNet chỉ là backbone; Torchvision GoogLeNet chỉ, không phải checkpoint BVLC Caffe; số streams và sharing + BiLSTM không xác nhận. [Model zoo](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html); Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, không được xếp hạng trực tiếp với FrameAcc. [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9); **Nguồn số liệu:** Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| Nayan et al. [Bài báo](https://doi.org/10.1007/s00138-024-01565-9) (không Seg) | 2024 | RGB+Flow+EHTD, GoogLeNet+BiLSTM | GoogLeNet ≈6.624904 /backbone | GoogLeNet 1.50 /224² /backbone | - | - | - | 69.40 | - | - | - | - | - | Không Seg; metric Levenshtein, không phải FrameAcc; cùng Nayan Table 12; Torchvision GoogLeNet chỉ, không phải checkpoint BVLC Caffe; số streams và sharing + BiLSTM không xác nhận. [Model zoo](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html); Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, không được xếp hạng trực tiếp với FrameAcc. [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9); **Nguồn số liệu:** Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| TY-Net [Bài báo](https://doi.org/10.1109/ACCESS.2023.3341702) | 2023 | MediaPipe skeleton+Transformer | TY-Net recognizer ≈5 ; Hands landmark 1.00/1.98 ; Palm192 1.133 | TY-Net 0.18 /recognizer; Palm192 0.354 /call; hand landmark - | ≈8.113 Full; ≈7.133 Lite | - | - | 60.46 | - | - | - | - | - | Online segmentation; Params/GFLOPs chưa tính MediaPipe; Params/GFLOPs tác giả là recognizer, thiếu MediaPipe.; TY-Net paper Section V-D nói ≈5M/0.18G nhận dạng; paper xác nhận dùng MediaPipe Hands trên IPN Hand nhưng không khóa model_complexity. Tổng minh họa ≈5+1.98+1.133=8.113M hoặc 5+1+1.133=7.133M, KHÔNG phải #Params nguyên văn của paper; detector Palm từ [Luxonis](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3), hand landmark từ [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/). Palm tracker chỉ chạy lại khi mất theo dõi.; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [TY-Net Sec. V-A](https://doi.org/10.1109/ACCESS.2023.3341702); **Nguồn số liệu:** Params (M), GFLOPs: do tác giả công bố; Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo; Params full pipeline (M): nhóm ước lượng từ các module tham chiếu, chưa xác minh toàn pipeline. |
| TMMF [Bài báo](https://doi.org/10.1109/TIP.2021.3108349) | 2021 | RGB+Flow, fusion theo thời gian | - | - | - | - | - | 68.12 | - | - | - | - | - | Số từ [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9); khác metric FrameAcc; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [Nayan Table 12 — TMMF](https://doi.org/10.1007/s00138-024-01565-9). |
| TD-Net [Bài báo](https://doi.org/10.1109/ICCAIS56082.2022.9990122) | 2022 | RGB→MediaPipe skeleton + detector/classifier | TD-Net: 1 detector + 2 recognizers ; Hands landmark 1.00/1.98 ; Palm192 1.133 | Palm192 0.354 /call; TD-Net networks + hand landmark - | - | - | - | 40.10 | - | - | - | - | - | Dùng skeleton; không phải FrameAcc; [Nguyen ICCAIS 2022](https://doi.org/10.1109/ICCAIS56082.2022.9990122) dùng MediaPipe Hands và 3 model TD-Net (1 detector, 2 classifiers). Phần MediaPipe Hands tham chiếu 2.133M (Lite+Palm) hoặc 3.113M (Full+Palm), **chưa gồm 3 TD-Net** => Params toàn pipeline vẫn '-'. [Hands](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md); [Luxonis Palm](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3); [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/).; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [TD-Net](https://doi.org/10.1109/ICCAIS56082.2022.9990122); **Nguồn số liệu:** Params (M): do tác giả công bố; Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| IPN original: ResNeXt+ResLight+SPyNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+Flow, original continuous | ResNeXt 47.56 + ResLight 0.908 + SPyNet 1.440 | - | 49.908 | - | 52 | 42.47 | - | - | - | - | - | Original IPN Table IX; không phải FrameAcc; cộng recognizer + ResLight detector (RGB 0.895M, RGB+Flow/Seg 0.908M) + Flow/Seg tùy hàng, từ IPN original Tables IV, V, VIII.; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [IPN official continuous benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html); **Nguồn số liệu:** Params (M): do tác giả công bố; Params full pipeline (M): nhóm cộng thông số các module được công bố. |
| IPN original: ResNeXt+ResLight+HarDNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB+Seg, original continuous | ResNeXt 47.56 + ResLight 0.908 + HarDNet 4.114 | - | 52.582 | - | 52 | 39.01 | - | - | - | - | - | Original Table IX; khác modality; cộng recognizer + ResLight detector (RGB 0.895M, RGB+Flow/Seg 0.908M) + Flow/Seg tùy hàng, từ IPN original Tables IV, V, VIII.; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [IPN official continuous benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html); **Nguồn số liệu:** Params (M): do tác giả công bố; Params full pipeline (M): nhóm cộng thông số các module được công bố. |
| IPN original: ResNeXt+ResLight [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020/21 | RGB, original continuous | ResNeXt 47.51 + ResLight 0.895 | - | 48.405 | - | 52 | 25.34 | - | - | - | - | - | Original Table IX; detector ResLight; cộng recognizer + ResLight detector (RGB 0.895M, RGB+Flow/Seg 0.908M) + Flow/Seg tùy hàng, từ IPN original Tables IV, V, VIII.; Số tại cột Accuracy (paper) là **Levenshtein Accuracy**, khác FrameAcc. [IPN official continuous benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html); **Nguồn số liệu:** Params (M): do tác giả công bố; Params full pipeline (M): nhóm cộng thông số các module được công bố. |

</details>

<details>
<summary>Phụ lục 3 — CTPGesture v1: Isolated Recognition và đối chứng ảnh dẫn xuất (bảng và chú thích đầy đủ)</summary>

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Params full pipeline (M) | GFLOPs full pipeline | N mẫu | Accuracy (%) | Accuracy@100% (%) | GT-prefix AUC | Ever-correct (%) | Chú thích / nguồn số liệu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB video, 16×224² | **6.959817** model | **29.0261** /clip | 6.959817 | 29.0261 /clip | 1.728 | 98.5532 | - | - | - | 9 lớp (0–8); 11 test videos; 869 no-gesture; 859 gesture. Accuracy@10/20/50/75, Top-2/3 và ECE-15 có output P1 trong 261028; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) — Gesture only (GT 1–8) | 2026 | RGB video, 16×224² | **6.959817** model | **29.0261** /clip | 6.959817 | 29.0261 /clip | 859 | 98.0210 | - | - | - | Đánh giá subset GT lớp 1–8 từ cùng checkpoint; loại GT no-gesture, nhưng model vẫn dự đoán được class 0. Không coi là mô hình thứ hai; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| **Zhang — BiLSTM–Transformer** [Bài báo](https://doi.org/10.3390/electronics15122578) | 2026 | **Skeleton:** YOLO11m-Pose→40-frame BiLSTM–Transformer | YOLO11m-Pose 20.9 + BiLSTM–Transformer 0.08641 | YOLO Pose 71.5 /frame; 40×71.5=2,860 (*pose stage only*) | ≈20.98641 | - | - | 98.91 | - | - | - | 10 video test; sequence classification; không cùng 11-video split; head ≠ pipeline; 40×71.5=2,860 chỉ tổng FLOPs pose nếu chạy đủ 40 ảnh 640²; chưa tính recognizer, không phải đo full pipeline; YOLO11m-Pose model zoo 20.9M/71.5G @640, Zhang Table 3 head 86.41K; **Nguồn số liệu:** Params (M): do tác giả công bố; Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo; GFLOPs, Params full pipeline (M): nhóm ước lượng từ các module tham chiếu, chưa xác minh toàn pipeline. |
| **FDMB-YOLOv11** [Bài báo](https://doi.org/10.1371/journal.pone.0357212) | 2026 | **Non-skeleton:** RGB frame, YOLO11n/MobileNetV4/BiFormer | **1.93** model | - | 1.93 detector | - | - | - | - | - | - | Ảnh dẫn xuất, task detection; mAP@0.5 khác Accuracy phân loại 1.728 clips; P/R/F1 là detection Table 3 model 7, không phải Macro-P/R/F1 classification; Table 6 normal-light có mAP/F1 không nhất quán Table 3 nên giữ Table 3 (mAP@0.5=97.17). [Nguồn Table 3](https://doi.org/10.1371/journal.pone.0357212.t003); **Nguồn số liệu:** Params (M), Params full pipeline (M): do tác giả công bố. |
| **Liu — Improved YOLOv8n** [Bài báo](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) | 2026 | **Non-skeleton:** RGB frame, YOLOv8n + attention | **1.8** model | **5.7** | 1.8 detector | 5.7 /ảnh | - | - | - | - | - | mAP theo paper, không mặc định là mAP@0.5; ảnh CTP + Internet, chia lại thành 16.000 ảnh; P/R là detection Table 2/3 của paper, không phải macro classification; F1 aggregate chưa tìm thấy trong bảng gốc. [Bài gốc](https://www.zjujournals.com/eng/EN/10.3785/j.issn.1008-973X.2026.08.007); **Nguồn số liệu:** Params (M), GFLOPs, Params full pipeline (M), GFLOPs full pipeline: do tác giả công bố. |
| **Assiri & Selim — Swin + CNN–BiLSTM** [Bài báo](https://doi.org/10.1016/j.asej.2025.103383) | 2025 | **Non-skeleton ứng viên:** RGB, Swin Transformer + CNN–BiLSTM | Swin-T ≈28.288354 model zoo | Swin-T 4.49 /224² | - | - | - | 99.25 | - | - | - | **Chưa xác minh đúng CTPGesture v1**; chỉ biết tên *Traffic Police Gesture Dataset*; không xếp hạng; chỉ Swin-T ImageNet của Torchvision, paper chưa xác nhận variant Swin; CNN–BiLSTM chưa đếm; không suy full. [Nguồn](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html); **Nguồn số liệu:** Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| **P3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, Pseudo-3D CNN | ≈65.3**P3D-199** | - | - | - | 1.565 | 92.21 | - | - | - | 9 test videos / 1.565 clips; **variant P3D chưa khóa**; baseline trong bài Xiong; 9 test videos/1.565 clips; các Params/GFLOPs là variant model zoo, **không** xác nhận checkpoint Xiong; **Nguồn số liệu:** Params (M): tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| **C3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, C3D | 78.4 MMAction2 | 38.5 /16×112² | - | - | 1.565 | 87.12 | - | - | - | 9 test videos / 1.565 clips; Params/GFLOPs model zoo không xác nhận checkpoint Xiong; 9 test videos/1.565 clips; các Params/GFLOPs là variant model zoo, **không** xác nhận checkpoint Xiong; **Nguồn số liệu:** Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |
| **3D ResNet** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | 2021 (baseline) | **Non-skeleton:** RGB video, 3D ResNet | 33.371472**R3D-18** | 40.70 /16×112² | - | - | 1.565 | 81.16 | - | - | - | 9 test videos / 1.565 clips; **không biết depth** của bản Xiong, số tham chiếu R3D-18; 9 test videos/1.565 clips; các Params/GFLOPs là variant model zoo, **không** xác nhận checkpoint Xiong; **Nguồn số liệu:** Params (M), GFLOPs: tham khảo từ model zoo, chưa xác nhận đúng biến thể của bài báo. |

</details>

<details>
<summary>Phụ lục 4 — CTPGesture v1: Continuous Gesture Recognition (bảng và chú thích đầy đủ)</summary>

| Phương pháp / nguồn | Năm | Đầu vào / kiến trúc | Params (M) | GFLOPs | Params full pipeline (M) | GFLOPs full pipeline | N video | Accuracy (paper, %) | FrameAcc global (%) | FrameAcc TB/video (%) | Temporal Overlap | Gesture Recall (%) | FP/min | Chú thích / nguồn số liệu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB video, **offline** 32/8 | **6.959817** model | 29.0261 /clip | 6.959817 | 29.0261 /clip | 11 | - | 83.9836 | 83.4973 | 0.653223 | 94.8035 | - | 11 videos / 67.356 frames; offline 32/8, có thể dùng future frame; F1@10/25/50, Edit, TP/FP/FN có output RAW P1 261028; ECE thuộc calibration isolated; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| VideoMamba R3 [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | 2026 | RGB offline 32/8 + post-processing | 6.959817 | 29.0261 /clip | 6.959817 | 29.0261 /clip | 11 | - | 83.9940 | - | 0.6590 | 94.45 | - | **Tuned trên official test**, không phải held-out; không dùng để khẳng định SOTA; **Nguồn số liệu:** Params full pipeline (M): đo trực tiếp từ notebook baseline; GFLOPs, GFLOPs full pipeline: nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn. |
| **RSC-GestureNet** [Bài báo](https://arxiv.org/abs/2608.02200) | 2026 | **Skeleton + RGB**, causal 32/4, 3 seeds | **0.815270** | - | **≥2.93** | - | - | **93.33±0.24** | - | - | - | - | - | **Params:** 0.815270M là **fusion recognizer** đếm chính xác theo [model.py](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) + [config hidden=96, 17 joints, RGB=12](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/configs/train.yaml). [Paper Sec. 3.1](https://www.researchgate.net/publication/411183058_RSC-GestureNet_Reliability-Aware_Selective_Causal_Recognition_of_Chinese_Traffic_Police_Gestures) xác nhận **MediaPipe Pose model_complexity=0 (BlazePose Lite)**; [BlazePose Lite 1.3M](https://www.mdpi.com/2076-3417/16/9/4497) + [PoseDetector 0.815M tham chiếu Qualcomm](https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation/blob/c4311a3344e055698541dc0d8fb310189284a4f7/README.md) + fusion 0.815270M = **≈2.930270M **. Đây là **cận dưới ước lượng theo các thành phần đã nhận diện**, chưa tính những nhánh pose/TCN dự phòng độc lập nếu triển khai theo Eq.(6), calibrator và sai khác checkpoint; **không được gọi 2.93M là exact toàn hệ thống**. **Accuracy:** [Table 1](https://arxiv.org/abs/2608.02200): RSC-GestureNet (Ours) **93.33±0.24%**, khác RSC-GestureNet-Fusion **91.38±0.38%**. Giao thức full-frame **causal** window 32/stride 4, mean±SD 3 seeds, 16.758 official test windows, **không trực tiếp đối chiếu với offline VideoMamba FrameAcc**. RGB descriptor là HSV histogram 12-D, không có CNN RGB riêng; **Nguồn số liệu:** Accuracy (paper, %): do tác giả công bố; Params (M): nhóm tính/ước lượng từ kiến trúc hoặc mã nguồn; Params full pipeline (M): nhóm ước lượng từ các module tham chiếu, chưa xác minh toàn pipeline. |

**Nguồn kiểm toán pipeline:** [IPN-Hand Tables IV/V/VIII](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition) (ResLight 0.895/0.908M, SPyNet 1.440M, HarDNet 4.114M và ResNeXt 47.51/47.56M); [Kausar](https://www.techscience.com/cmc/v89n3/69295/html) + [ResNet-34 Torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html) (encoder loại FC); [Nayan](https://doi.org/10.1007/s00138-024-01565-9) + [GoogLeNet model zoo](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html); [Assiri & Selim](https://doi.org/10.1016/j.asej.2025.103383) + [Swin-T model zoo](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html); [Zhang Table 3](https://doi.org/10.3390/electronics15122578) + [Ultralytics YOLO11m-Pose](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md); [FDMB Table 3](https://doi.org/10.1371/journal.pone.0357212.t003); [Liu Tables 2–3](https://www.zjujournals.com/eng/EN/10.3785/j.issn.1008-973X.2026.08.007); [RSC Table 1](https://arxiv.org/abs/2608.02200). Các ô total pipeline bằng `-` không được thay bằng Params một backbone riêng.

**Khảo sát MediaPipe và tổng Params pipeline (29/10/2026):** MediaPipe **Hands** khác **Pose**, không áp cùng một số Params cho các bài. [MediaPipe Hands chính thức](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md) gồm 2 mô hình *palm detector* và *hand landmark*; [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/) ghi 1.00M/1.98M Params cho Hand Lite/Full; [Luxonis 192×192](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3) đo 1.133M Params và 0.354 GFLOPs cho một biến thể palm detector. Vì chưa xác nhận tác giả dùng đúng *palm model* Luxonis hoặc `model_complexity`, các giá trị cộng vào Gil-Martín và TY-Net đều là **ước lượng có điều kiện, không phải số liệu tác giả công bố**. `model_complexity` mặc định MediaPipe Hands là 1 (Full) nếu tác giả không ghi đè, nhưng *mặc định phần mềm không chứng minh được cấu hình checkpoint bài báo*.

**MediaPipe Pose:** [BlazePose gốc](https://www.alphaxiv.org/abs/2006.10204) báo Lite ≈1.3M và Full ≈3.5M Params tham chiếu. [MediaPipe Pose](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) còn có Heavy và mạng person/pose detector, nên 1.3/3.5M chỉ phản ánh thành phần pose landmark theo kiến trúc tham chiếu, không phải full Pose pipeline. RSC dùng pose features đã trích, không phát hành đầy đủ cấu hình detector/checkpoint gốc; do đó **cột full Params vẫn là '-';** chỉ ghi cộng tham chiếu thành phần trong ghi chú. TD-Net có thêm detector và hai bộ phân loại TD-Net chưa xác minh chính xác Params, nên chỉ bổ sung chi phí MediaPipe tham chiếu. **Không chuyển số Model Size (MB), file .tflite (MB) hay FLOPs mỗi lần chạy sang #Params hoặc GFLOPs toàn chuỗi khi thiếu định nghĩa.**

</details>

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
