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
7. [**TMMF: Temporal Multi-modal Fusion for Single-Stage Continuous Gesture Recognition**](https://arxiv.org/html/2011.04945v2) (2021) — Single-stage continuous gesture recognition bằng RGB + Optical Flow; [IEEE DOI](https://doi.org/10.1109/TIP.2021.3108349). **Đã kiểm chứng:** [Table III trong bản toàn văn arXiv](https://arxiv.org/html/2011.04945v2) báo Levenshtein Accuracy **68.12%** trên IPN-Hand.
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

### Table 1 — IPN-Hand: Isolated Gesture Recognition

| Method / Reference | Input / Architecture | Params (M) | GFLOPs | Accuracy (%) | GT-prefix AUC | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB 16×224² | 6.9608 | 29.0261 | 84.65 | 0.72399 | VideoMamba; RGB 16 frames; GT-prefix offline. Chi phí theo clip, không phải toàn hệ thống. |
| CoboGesture + **VideoMAEv2** [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, ViT-L | ≈303.8 (recognizer) | 596.8327 | 84.59 | - | Khác cách lấy mẫu và tiền xử lý so với VideoMamba. |
| CoboGesture + VideoMAE [Cobo](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, ViT-L | ≈303.8 (recognizer) | 596.8327 | 78.81 | - | Biến thể VideoMAE đời trước, cùng bài CoboGesture. |
| Kausar et al. [Nguồn](https://doi.org/10.32604/cmc.2026.084514) | RGB + Flow + Seg + Transformer | ≈21.284672 (encoder tham chiếu) | ResNet-34 3.66/ảnh (tham chiếu) | 92.34 | - | Chưa khóa split; tham số chỉ phản ánh ResNet-34 encoder. |
| Gil-Martín et al. [Nguồn](https://doi.org/10.5220/0013053500003890) | MediaPipe landmarks+LSTM | 0.058614 (LSTM); ≈2.192/3.172 (conditional full) | Palm192 0.354 /call; hand landmark + LSTM: - | 84.66±1.09; 86.38±2.03 | - | **Hai giao thức:** 10-fold CV / 37 train–13 test; 0,058614M chỉ LSTM. Full pipeline **ước lượng có điều kiện**: LSTM 0.058614M + Palm 1.133M + Hands landmark Lite 1.00M (hoặc Full 1.98M) = 2.191614/3.171614M. [Palm reference](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3); [Hands reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/). Chưa xác minh đúng checkpoint trong bài Gil-Martín. |
| IPN original: ResNeXt-101 RGB+Flow [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+SPyNet, 32f | 47.56 + 1.440 | - | 86.32 | - | RGB + optical flow; khác dữ liệu đầu vào và số khung. |
| IPN original: ResNeXt-101 RGB+Seg [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+HarDNet, 32f | 47.56 + 4.114 | - | 84.77 | - | RGB + segmentation; khác dữ liệu đầu vào và số khung. |
| IPN original: ResNeXt-101 RGB [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB, 32f | 47.51 | - | 83.59 | - | Baseline RGB; 32 frames, khác thiết lập 16 frames. |

**Lưu ý:** Gil-Martín có **hai giao thức độc lập** (10-fold CV và 37 train/13 test); hai Accuracy trong cùng ô theo đúng thứ tự này. Những số Params chỉ là *recognizer* hoặc backbone được ghi rõ, không nên so trực tiếp với chi phí toàn pipeline.

### Table 2 — IPN-Hand: Continuous Gesture Recognition

| Method / Reference | Input / Protocol | Params (M) | Accuracy (%) | Accuracy Metric | Overlap | Gesture Recall (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB video, offline 64/16 | 6.9608 | 83.4889 | FrameAcc (mean/video; measured) | 0.491110 | 78.8995 | RAW offline 64/16; không phải đánh giá online. Accuracy được tính là trung bình FrameAcc từng video theo notebook, không phải Levenshtein Accuracy. |
| VideoMamba R3 [Notebook](../261007/videomamba_man_configsBenchmark.ipynb) | RGB, post-processing 64/16 | 6.9608 | 83.5249 | FrameAcc (mean/video; measured) | 0.507705 | 76.2844 | R3 chọn ngưỡng trên test; không dùng để khẳng định hơn baseline. FrameAcc mean/video đo trên bộ 52 video; R3 hậu xử lý, threshold đã chọn trên test. |
| CoboGesture + VideoMAEv2 [Nguồn](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, offline 64/30 | ≈303.8 (recognizer) | 83.00 | Frame-wise Accuracy (paper) | 0.510 | - | Offline 64/30; cách tính overlap cần đối chiếu. Frame-wise Accuracy được báo trực tiếp trong [CoboGesture Table 7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction), không phải Levenshtein Accuracy; chưa xác nhận cách gộp theo video tương đương VideoMamba. |
| CoboGesture + VideoMAE [Nguồn](https://doi.org/10.1109/ACCESS.2026.3664027) | RGB video, offline 64/30 | ≈303.8 (recognizer) | 80.00 | Frame-wise Accuracy (paper) | 0.370 | - | Offline 64/30; cách tính overlap cần đối chiếu. Frame-wise Accuracy được báo trực tiếp trong [CoboGesture Table 7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction), không phải Levenshtein Accuracy; chưa xác nhận cách gộp theo video tương đương VideoMamba. |
| Nayan et al. [Nguồn](https://doi.org/10.1007/s00138-024-01565-9) | RGB + Flow + EHTD + Seg | ≈6.624904 (per GoogLeNet, ref.) | 73.71 | Levenshtein Accuracy (paper) | - | - | Giá trị bài báo là **Levenshtein Accuracy**, không phải FrameAcc. [Torchvision GoogLeNet](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html) công bố 6.624904M Params/1.50 GFLOPs cho **một backbone tham chiếu**; chưa tính số streams thực tế, sharing và BiLSTM, do đó không phải tổng Params được bài báo công bố. |
| Nayan et al. [Nguồn](https://doi.org/10.1007/s00138-024-01565-9) (không Seg) | RGB + Flow + EHTD | ≈6.624904 (per GoogLeNet, ref.) | 69.40 | Levenshtein Accuracy (paper) | - | - | Không dùng segmentation; giá trị là **Levenshtein Accuracy**. [Torchvision GoogLeNet](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html) công bố 6.624904M Params/1.50 GFLOPs cho **một backbone tham chiếu**; chưa tính số streams thực tế, sharing và BiLSTM, do đó không phải tổng Params được bài báo công bố. |
| TY-Net [Nguồn](https://doi.org/10.1109/ACCESS.2023.3341702) | Skeleton + Transformer | ≈5.0 (recognizer, paper) | 60.46 | Levenshtein Accuracy (paper) | - | - | Skeleton; kết quả online theo Levenshtein, khác FrameAcc. ≈5M thuộc TY-Net recognizer theo paper, chưa tính MediaPipe hand detector/landmark; không suy ra full pipeline nếu chưa xác nhận biến thể MediaPipe. |
| TMMF [Full text](https://arxiv.org/html/2011.04945v2) | RGB + Optical Flow; single-stage | **≥47.016 M (2 ResNet50 backbones, tham chiếu)** | 68.12 | Levenshtein Accuracy (paper) | - | - | **Params chưa được tác giả công bố cho toàn TMMF**. Ước lượng tham chiếu cho **riêng 2 feature extractors độc lập**, theo Section III-A/III-E: mỗi ResNet50 [Torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50) có 25.557032M, loại lớp FC ImageNet 2.049M còn 23.508032M; 2 streams × 23.508032M = **47.016064M**. Đây là **cận dưới có điều kiện theo kiến trúc tham chiếu**, **chưa cộng** UFM (12 residual layers/stream), Fusion và MFM (10 layers), **không phải số đo checkpoint hoặc full pipeline chính thức**. Table III của [bài gốc](https://arxiv.org/html/2011.04945v2) báo **68.12% Levenshtein Accuracy** trên IPN-Hand (không phải FrameAcc); [DOI IEEE](https://doi.org/10.1109/TIP.2021.3108349). |
| TD-Net [Nguồn](https://doi.org/10.1109/ICCAIS56082.2022.9990122) | Skeleton + TD-Net | 2.133–3.113 (MediaPipe refs only) | 40.10 | Levenshtein Accuracy (paper) | - | - | Skeleton; kết quả theo Levenshtein, khác FrameAcc. Đây chỉ là phép cộng **tham chiếu có điều kiện** Palm detector 1.133M ([Luxonis](https://models.luxonis.com/luxonis/mediapipe-palm-detection/9531aba9-ef45-4ad3-ae03-808387d61bf3)) + hand landmark Lite 1.00M hoặc Full 1.98M (nguồn [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244600/)); **không bao gồm** 3 TD-Net detector/classifier và không xác minh checkpoint tác giả. Không được xem là Params chính thức của TD-Net. |
| IPN original: ResNeXt+ResLight+SPyNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+Flow, original continuous | 47.56 + 0.908 + 1.440 | 42.47 | Levenshtein Accuracy (paper) | - | - | Benchmark gốc; Levenshtein Accuracy, không phải FrameAcc. |
| IPN original: ResNeXt+ResLight+HarDNet [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB+Seg, original continuous | 47.56 + 0.908 + 4.114 | 39.01 | Levenshtein Accuracy (paper) | - | - | Benchmark gốc với segmentation; Levenshtein Accuracy. |
| IPN original: ResNeXt+ResLight [Nguồn](https://gibranbenitez.github.io/IPN_Hand/Results.html) | RGB, original continuous | 47.51 + 0.895 | 25.34 | Levenshtein Accuracy (paper) | - | - | Benchmark gốc chỉ RGB; Levenshtein Accuracy. |

**Accuracy Metric:** `FrameAcc (mean/video; measured)` là số nhóm đo từ VideoMamba, `Frame-wise Accuracy (paper)` là số CoboGesture báo tại Table 7, còn `Levenshtein Accuracy (paper)` đánh giá chuỗi nhãn cử chỉ. **Không được xếp hạng trực tiếp các giá trị ở ba nhóm metric này** khi chưa chuẩn hóa cùng một protocol. Các paper IPN gốc dùng Levenshtein cho continuous.

### Table 3 — CTPGesture v1: Isolated Gesture Recognition and Related Image-Based Baselines

| Method / Reference | Input / Architecture | Params (M) | Accuracy (%) | Task Type | Notes |
| --- | --- | --- | --- | --- | --- |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB video, 16×224² | 6.959817 | 98.5532 | Phân loại video | VideoMamba 9 lớp, 11 test videos; 1.728 GT clips. |
| **VideoMamba-Tiny** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) — Gesture only (GT 1–8) | RGB video, 16×224² | 6.959817 | 98.0210 | Phân loại video | Subset 859 gesture GT từ cùng mô hình 9 lớp. |
| **Zhang — BiLSTM–Transformer** [Nguồn](https://doi.org/10.3390/electronics15122578) | Skeleton; 40 frames | 20.9 (pose) + 0.08641 (head) | 98.91 | Video skeleton | Skeleton; 10 test videos; split khác VideoMamba. |
| **FDMB-YOLOv11** [Nguồn](https://doi.org/10.1371/journal.pone.0357212) | RGB frame; YOLO11n | 1.93 | - | Detection ảnh | **Detection ảnh**; mAP không thể điền vào Accuracy video. |
| **Liu — Improved YOLOv8n** [Nguồn](https://doi.org/10.3785/j.issn.1008-973X.2026.08.007) | RGB frame; YOLOv8n | 1.8 | - | Detection ảnh | **Detection ảnh**; dùng tập ảnh xử lý lại. |
| **Assiri & Selim — Swin + CNN–BiLSTM** [Nguồn](https://doi.org/10.1016/j.asej.2025.103383) | RGB; Swin + CNN–BiLSTM | ≈28.288354 (Swin-T tham chiếu) | 99.25 | Video; chưa khóa dataset | Chưa xác minh sử dụng đúng CTPGesture v1. |
| **P3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, Pseudo-3D CNN | ≈65.3 (P3D-199 tham chiếu) | 92.21 | Phân loại video | Baseline trong Xiong, 9 video / 1.565 clips; variant tham chiếu. |
| **C3D** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, C3D | 78.4 (tham chiếu) | 87.12 | Phân loại video | Baseline Xiong; cấu hình model zoo chỉ tham khảo. |
| **3D ResNet** [Xiong Table 3](https://doi.org/10.3390/electronics10050551) | RGB video, 3D ResNet | 33.371472 (R3D-18 tham chiếu) | 81.16 | Phân loại video | Baseline Xiong; độ sâu mạng gốc chưa xác minh. |

**Lưu ý:** Các công trình **detection trên ảnh** (FDMB-YOLOv11, Liu) không có Accuracy phân loại video tương ứng. Bản Assiri & Selim chưa xác minh dùng đúng CTPGesture v1; các baseline Xiong có split khác.

### Table 4 — CTPGesture v1: Continuous Gesture Recognition

| Method / Reference | Input / Protocol | Params (M) | Reported Accuracy (%) | Global Frame Accuracy (%) | Mean Frame Accuracy / Video (%) | Overlap | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **VideoMamba Raw** [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB video, offline 32/8 | 6.959817 | - | 83.9836 | 83.4973 | 0.653223 | RAW offline 32/8; có sử dụng frame tương lai. |
| VideoMamba R3 [Notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb) | RGB offline 32/8 + post-processing | 6.959817 | - | 83.9940 | - | 0.6590 | R3 tinh chỉnh trên official test; không phải held-out. |
| **RSC-GestureNet** [Nguồn](https://arxiv.org/abs/2608.02200) | Skeleton + RGB; causal 32/4 | 0.815270 (fusion); ≈2.930 (conditional) | **93.33±0.24** | - | - | - | Causal 32/4, 3 seeds; **không so trực tiếp** với FrameAcc offline. Fusion recognizer 0.815270M được đếm từ [official source code](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py). Tham chiếu cho pipeline nếu dùng **BlazePose Lite landmark 1.3M + PoseDetector Qualcomm 0.815M** là 0.815270+1.3+0.815 = **≈2.930270M**; phần Pose chỉ là variant tham chiếu, chưa xác minh exact checkpoint, các nhánh khác chưa tính. Không báo ≈2.93M như số Params tác giả công bố toàn hệ thống. |

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

**CTPGesture gesture-only (859 đoạn):** nhóm đã có **Macro-F1 = 98.2363%** và **Balanced Accuracy = 98.0030%** từ cùng checkpoint 9 lớp. Không đưa tập con này vào bảng Top-K vì chưa có các kết quả Top-2/Top-3, Ever-correct và First-correct riêng.

Chỉ số *first-correct* và *first-stable* là tỷ lệ chiều dài đoạn theo giao thức GT-prefix; không phải độ trễ tính bằng giây.

### 5.3. Isolated — Calibration và minh chứng keyframe

| Mô hình / tập test | ECE-15 | Brier | NLL | Keyframe đã xuất |
| --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba-Tiny | 0.029848 | 0.197447 | 0.420977 | 1.101 JPEG: 932 đúng / 169 sai |
| CTP · VideoMamba (9 lớp) | 0.005987 | 0.021505 | 0.051804 | 1,728/1,728 JPEG; 1,703 đúng; 25 sai; 0 lỗi |

ECE, Brier và NLL chỉ đánh giá **calibration ở isolated** cho các tập có output xác minh. **Subset CTPGesture gesture-only gồm 859 đoạn dùng chung bộ 1.728 ảnh JPEG đã xuất từ mô hình 9 lớp**, chưa có ECE/Brier/NLL được tính riêng. Số keyframe trong bảng là tổng trường hợp có log xuất hình đúng/sai, **không phải số video độc lập**. Xem ví dụ trong [báo cáo 28/10](../261028/Readme.md).

### 5.4. Continuous — FrameAcc, Macro-F1 và chất lượng phân đoạn

| Mô hình / tập test | FrameAcc global (%) | FrameAcc TB/video (%) | Macro-F1 (%) | Seg F1@10 (%) | Seg F1@25 (%) | Seg F1@50 (%) | Edit (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba RAW | 83.7308 | 83.4889 | 64.4012 | 79.4544 | 77.5789 | 69.9062 | 74.3831 |
| CTP · VideoMamba RAW | 83.9836 | 83.4973 | 80.6135 | 96.0472 | 94.7493 | 85.5457 | 93.4968 |

**Kết quả R3 đã có từ notebook baseline:** IPN-Hand R3 có **FrameAcc TB/video = 83.5249%**; CTPGesture R3 có **FrameAcc global = 83.9940%** và **Macro-F1 = 80.6254%**. Đây là chỉ số baseline, **không phải metric Segment F1/Edit của P1**. Nhánh R3 đã được tinh chỉnh trên tập test; không dùng kết quả này như phép đánh giá held-out độc lập.

FrameAcc theo video khác cách tổng hợp FrameAcc trên tất cả frame.

### 5.5. Continuous — Ghép đoạn, sai lệch biên và keyframe

| Mô hình / tập test | Precision@50 (%) | Recall@50 (%) | Onset MAE (s) | Offset MAE (s) | FP/min | Temporal overlap | Keyframe đã xuất |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPN-Hand · VideoMamba RAW | 65.8635 | 74.4777 | 0.440922 | 0.481819 | 3.596927 | 0.491110 | 1.526 JPEG: 820 ghép đúng / 706 lỗi |
| CTP · VideoMamba RAW | 86.7225 | 84.4005 | 0.290115 | 0.445517 | 1.483164 | 0.653223 | 970/970 JPEG; 725 matched; 245 lỗi; 0 lỗi xuất |

**Các kết quả R3 đã có từ notebook baseline:** IPN-Hand R3 đạt **Temporal overlap = 0.507705**; CTPGesture R3 đạt **Temporal overlap = 0.658979**. Cả hai mới có **mã tạo timeline GT/Pred từ 16 frame**; chưa có bộ ảnh P1 xuất riêng cho R3. Không chuyển Precision/Recall, sai lệch biên hay FP/min từ RAW sang R3.

**FP/min** trong các notebook P1 RAW được tính từ đoạn dự đoán không ghép được mỗi phút (ngưỡng tIoU 0,50), không mặc định trùng định nghĩa FP/min của mọi bài báo. Các sai số onset/offset MAE tính trên những đoạn đã ghép đúng.

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
