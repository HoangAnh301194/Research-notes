# Báo cáo nghiên cứu ngày 22/10/2026

## A. Công việc đã làm 
- Tổng quan lại toàn bộ kết quả đã làm 
- Khảo sát các bài báo :
    - Cùng sử dụng IPN Hand : none Skeleton 
    - Cùng sử dụng CTP Gesture v1 : 1-2 bài SOTA có skeleton và ~5 bài RGB input 


## B. Khó khăn 

## C. Công việc tiếp theo

---

# PHỤ LỤC CẬP NHẬT NGÀY 27/10/2026 — KIỂM ĐỊNH KẾT QUẢ VÀ 4 BẢNG ĐỐI CHIẾU SOTA

> **Mục đích:** Bổ sung báo cáo ngày 22/10, phục vụ tổng hợp tài liệu gửi thầy. Phụ lục được lập bằng việc đọc source code, cấu hình và output đã lưu trong notebook của repository; đối chiếu các bảng/công thức ở bài báo gốc. **Không chạy lại training/inference và chưa xác nhận checkpoint bằng tái lập độc lập.** Không coi sự chênh lệch giữa hai giao thức khác nhau là bằng chứng một phương pháp vượt SOTA.

## D. Phạm vi, notebook và quy ước chứng cứ

### D.1. Notebook đã rà soát

1. [IPN-Hand evaluation + complexity (261007)](../261007/videomamba_man_configsBenchmark.ipynb): isolated cell 12, continuous cell 18, hậu xử lý cell 20, GFLOPs/latency cells 22–30.
2. [CTPGesture v1 final evaluation (261015)](../261015/VideoMambaTinyCTPGestureFinal.ipynb): data split cell 10, checkpoint cell 12, isolated cell 16, continuous cell 22, hậu xử lý cell 24, complexity cell 26.
3. [CTPGesture initial evaluation](../261007/evaluation.ipynb): single-view, 12-view; **khác** giao thức của final isolated notebook.
4. [CTPGesture training notebook](../KaggleNotebook/videomamba_ctpgesture_v1_train.ipynb): tách train/val/test theo video ID, kiểm tra tập không giao nhau; bản trong repository không có execution outputs để kiểm chứng toàn bộ quá trình training.
5. [VideoMAE v1/v2 architecture comparison](../KaggleNotebook/videomamba-vs-videomaev1-v2.ipynb): benchmark kiến trúc, **không phải** thí nghiệm độ chính xác VideoMAE trên chính các dataset với cùng code training.
6. [EgoEvGesture notebooks](../KaggleNotebook/EgoEvGesture_Notebooks): xây dựng/kiểm tra split và huấn luyện resume; chưa thấy kết quả kiểm thử có execution outputs trong các notebook đã đọc.
7. [Notebook training CTP lỗi ở epoch 29](../261007/train_failed_echoch29.ipynb): file có trong repo nhưng kết nối GitHub chưa trả được nội dung đầy đủ của file lớn; không dùng file này để xác nhận chi tiết lịch sử training.

**Bằng chứng ký hiệu trong bảng:** `[O]` số đo/output từ notebook nhóm; `[P]` số được bài báo/benchmark gốc công bố; `[E]` số ước lượng kiến trúc hoặc đo lại bằng code nhóm (không phải số do công trình đối chứng công bố); `—` không thấy số có thể xác nhận.

**Mức độ đối chiếu:**
- **A:** Cùng dataset và metric tương đồng, còn phải xác nhận chi tiết split/preprocessing trước khi gọi là fair comparison.
- **B:** Cùng tên dataset nhưng khác input, subset, split, aggregation hoặc test protocol; chỉ đối chiếu tham khảo.
- **C:** Khác định nghĩa tác vụ/metric, hoặc phương pháp chỉ là nghiên cứu liên quan; **không so sánh thứ hạng**.

**Quy tắc:** Không điền Params/GFLOPs của biến thể kiến trúc khác như thể đó là số đo gốc. Không tính latency trên Tesla T4 rồi so trực tiếp với GPU được tác giả dùng. Không dùng từ "SOTA vượt trội" nếu chưa khớp protocol, tính biến thiên và tập test.

## E. BỐN BẢNG ĐỐI CHIẾU

**Cách đọc nguồn ngay trong bảng:** `Notebook` = số tự đo; `Paper/Table` = số tác giả công bố; `model zoo/reference` = thông số của biến thể tham khảo (không đảm bảo trùng checkpoint benchmark). Các số ước lượng vẫn giữ nhãn †/[E] và ghi chú chi tiết dưới bảng.

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

**Không đặt Levenshtein Accuracy cùng cột với Frame-wise Accuracy.** Hai phép đo thuộc hai benchmark khác nhau, nên Bảng 2 được chia thành hai phần có dữ liệu thực và không tạo hàng toàn dấu `—`.

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

`*` Toàn pipeline suy từ classifier Table IV + detector ResLight Table VIII (0.895M RGB, 0.908M RGB+Flow/Seg) + preprocessing SPyNet 1.440M/HarDNet 4.114M theo Table V. Đây là **tổng tham số mạng theo sơ đồ chức năng**, không phải model state_dict gộp do tác giả phát hành; phần chi phí tính toán pipeline có thể khác do sampling/đệm. VideoMamba hiện chưa tính **Levenshtein** nên **không đặt vào Bảng 2B**. Các giá trị latency lấy từ benchmark IPN gốc trên GTX 1080Ti, không so thẳng với phép đo T4.

**Đối chiếu chủ đạo:** FrameAcc của Cobo là **mean-per-video**, dùng **83.4889% Raw / 83.5249% R3**, không thay bằng Global FrameAcc 83.7308%/83.7671%. R3 được chọn bằng test-label tuning nên chưa đủ điều kiện công bố như test độc lập. Không tuyên bố mô hình vượt Cobo nếu chưa đồng nhất class mapping/overlap matching.


### Bảng 3 — CTPGesture v1: Isolated / clip classification

**Giao thức VideoMamba nhóm:** 11 video test/1.728 segments, class 0–8, uniformly sample 16 RGB frames. **Các baseline Xiong (2021) dùng 9 video test/1.565 segments; không xếp hạng trực tiếp**. Khác IPN, nhiều công trình CTP không công bố Macro-F1 và Params đúng biến thể, nên bảng này ưu tiên **Accuracy + complexity có bằng chứng**; Macro-F1 riêng của nhóm giữ trong chú thích, không lấp bằng giá trị không cùng protocol.

| Method | Năm | Input | **Params recognizer (M)** | **Params full pipeline (M)** | **GFLOPs/clip** | **Accuracy (%)** | Bằng chứng/thời gian | Nguồn |
|---|---:|---|---:|---:|---:|---:|---|---|
| **VideoMamba-Tiny (OURS, 9 lớp)** | 2026 | RGB, 16×224² | **6.959817** | **6.959817** | **29.0261†** | **98.5532** | Macro-F1=**97.9590%**; 1.728 clips | [CTP notebook cell 16/26](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| GSE + MD-GCN (Xiong) | 2021 | OpenPose→skeleton, 80f | **—** | **—** | — | **98.95** | Model Size = **11.7 M** (*đơn vị chưa xác thực*); Inference Time = **3.0 s** (theo Table 1, đơn vị đo thao tác chưa rõ) | [Xiong Table 1/3](https://www.mdpi.com/2079-9292/10/5/551) |
| P3D (Qiu; đối chứng Xiong) | 2021 | RGB | **≈65.3‡** | — | — | **92.21** | P3D-199 reference, **không xác minh checkpoint Xiong** | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [P3D gốc](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf) |
| C3D (Tran; đối chứng Xiong) | 2021 | RGB | **78.4‡** | — | **38.5‡** | **87.12** | C3D MMAction2: **16×112²**, model family reference | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [C3D model zoo](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md) |
| 3D ResNet (Hara; đối chứng Xiong) | 2021 | RGB | **33.371472‡** | — | **40.697‡** | **81.16** | Torchvision R3D-18: **16×112²**, reference | [Xiong](https://www.mdpi.com/2079-9292/10/5/551) · [R3D-18 model zoo](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html) |

`†` FLOPs VideoMamba từ bộ đếm thủ công (1 MAC = 1 FLOP-equivalent), không thể so thẳng với GFLOPs `‡` lấy từ model zoo cấu hình khác. `‡` **Số tham khảo của cùng họ kiến trúc, KHÔNG phải số đo của biến thể mà Xiong dùng**. Không được gọi là benchmark công bằng. 11.7 M của Xiong là cột *Model Size*, **không xác nhận đó là 11.7 triệu tham số**; cần định nghĩa MB/Mparams hoặc checkpoint trước khi điền vào Params.

**VideoMamba theo subset:** Gesture-only (859 GT segments, bỏ 869 class-0 segments) Accuracy **98.0210%**, Macro-F1 **98.2363%**; không đánh đồng với 9-class Accuracy.

#### Nghiên cứu 2025–2026 cùng chủ đề, nhưng **không phải isolated benchmark tương đương**

| Method | Năm | Input | **Params** | **Result được tác giả công bố** | **Runtime công bố** | Lý do không xếp với VideoMamba | Nguồn |
|---|---:|---|---|---|---|---|---|
| YOLO11m-Pose → BiLSTM–Transformer | 2026 | RGB→Pose | **0.08641M recognition head; ≈20.99M pipeline‡** | **98.91% sequence recognition** | **1.3299 s/sequence head** (chưa tính pose extraction) | 10 test videos, sequence-to-one | [Zhang 2026 Table 2/3](https://www.mdpi.com/2079-9292/15/12/2578) |
| NTPGR (EPFFNet + MSNet) | 2025 | RGB→Pose | **—** (thiếu exact checkpoint) | **97.56% gesture-instance** | **0.76 s average response** trên TITAN XP | Khác split/đơn vị đánh giá | [Xiao 2025](https://www.nature.com/articles/s41598-025-02833-y) |

`‡` YOLO11m-Pose ~20.9M lấy từ model zoo Ultralytics, cộng head 86.41K; tổng **chỉ là suy luận theo cấu hình**, không phải full-pipeline parameter do Zhang công bố. Không đưa NTPGR 0.76s cạnh 1.3299s hoặc VideoMamba 50ms để xếp tốc độ: các phép đo không cùng phạm vi tác vụ/GPU.


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


## F. KIỂM TRA KẾT QUẢ NOTEBOOK: ĐÃ ĐÚNG Ở ĐÂU, CHƯA ĐỦ Ở ĐÂU

| Hạng mục | Kết quả audit | Độ tin cậy và hành động |
|---|---|---|
| IPN isolated Top-1 84.65%, 1.101 samples | Công thức trung bình đúng `correct.mean()`, output đã lưu | **Xác minh từ code + output;** đối chiếu 1.101 mẫu với Cobo |
| IPN continuous Raw mean FrameAcc 83.4889% / global 83.7308% | Hai metric khác nhau, notebook báo cáo cả hai | **Đúng công thức nội bộ**; công bố mean per-video khi đối chiếu Cobo |
| IPN R3 83.5249%, overlap 0.507705 | Chọn `T_AVG=0.7 s`, `MIN_GESTURE=1.0 s` theo metric trên **test** | **Chỉ exploratory/test-tuned**, không xem là independent held-out result |
| CTP isolated 98.5532%, Macro-F1 97.9590% | 1.728 segments, 11 video; `sklearn.metrics`, checkpoint `strict=True` | **Khớp code + output**, không đồng nhất split Xiong |
| CTP continuous Raw Accuracy 83.9836%, Macro-F1 80.6135% | 67.356 frames, 8.392 windows, 11 videos | **Khớp code + output**, offline full-window aggregation |
| CTP R3 Accuracy 83.9940%, Macro-F1 80.6254% | Dò `MIN_GESTURE=1.0 s` theo Global Macro-F1 trên **test** | **Test-tuned**, cải thiện rất nhỏ, recall giảm |
| Param IPN 6.9608M; CTP 6.959817M | Khác do số lớp của classification head (14 so với 9) | **Giữ tách theo dataset**, không làm tròn hai giá trị thành cùng count chính xác |
| GFLOPs VideoMamba 29.0261 [E] | Hàm đếm thủ công tính các toán tử Mamba, quy ước `1 MAC = 1 FLOP-equivalent` | **Ước lượng**, không đồng nhất với phép profiling `fvcore` |
| CTP training split | Code assert không giao nhau giữa video ID train/val/test | **Kiểm tra cấp video ID**; cần lưu manifest và mã băm mẫu cho reproducibility |
| CTP 12-view 98.50% trong evaluation cũ | Khác số views và cách sampling so với 98.5532% uniform sampling của final | **Không gộp** chung một hàng hoặc một protocol |
| Model-only latency khoảng 48–51 ms/clip trên T4 | CUDA sync, FP32, model-only; wall throughput CTP 8.67 windows/s | **Không phải** độ trễ phát lệnh tính từ lúc gesture bắt đầu |
| EgoEvGesture | Notebook chuẩn bị split/train resume, chưa có execution outputs | **Chưa công bố kết quả**, tránh điền bảng đoán |

**GFLOPs phải giải trình:** bản profiling `fvcore` ở notebook training CTP ghi khoảng **17.19 G/view**, trong khi bộ đếm thủ công cuối cùng cho cùng VideoMamba khoảng **29.0261 G/clip**. `fvcore` có thể thiếu custom Mamba ops; chưa đủ căn cứ coi bất kỳ số nào là *ground-truth FLOPs*. Cần công bố cùng quy ước, input, danh sách toán tử được đếm và kiểm tra unsupported ops. Model ViT-L của VideoMAE v1/v2 được nhóm dựng cho benchmarking kiến trúc, **không** chứng minh Accuracy của VideoMAE trên dataset nhóm.

**Chính sách chọn ngưỡng:** Cả IPN và CTP hiện có R3 được dò trực tiếp từ nhãn test. Để sửa, cần tìm `T_AVG`, `MIN_GESTURE`, smoothing trên **validation** rồi khóa cấu hình và đánh giá test. Có thể thử trên probability cache mà **không cần chạy GPU**, nếu cache cho đúng validation còn lưu được.

**Cần phân biệt offline / online:** Hàm `windows_to_frame_probs()` cộng xác suất vào **toàn bộ** `[start, end]` của cửa sổ, tạo ra dự đoán frame-level có nhìn trước tương lai; đây là đánh giá phân đoạn offline. Muốn đối chiếu RSC cần đầu ra causal chỉ dùng dữ liệu đến thời điểm ra lệnh, thiết kế lại evaluation và tính Online-F1, TTC, Early@10.

## G. DANH SÁCH NGHIÊN CỨU 2023–2026 VÀ GIỚI HẠN

| Công trình | Năm | Liên quan | Quyết định |
|---|---:|---|---|
| [CoboGesture, IEEE Access](https://doi.org/10.1109/ACCESS.2026.3664027) | 2026 | IPN isolated/continuous, RGB VideoMAE v1/v2 | **Giữ bảng 1–2, đối chứng chính** |
| [RSC-GestureNet, arXiv / PRCV](https://arxiv.org/abs/2608.02200) | 2026 | CTPGesture v1 causal full-frame, pose/RGB fusion | **Giữ bảng 4, nhưng khác protocol** |
| [NTPGR, Scientific Reports](https://www.nature.com/articles/s41598-025-02833-y) | 2025 | Traffic police, pose + spatiotemporal fusion | Nghiên cứu liên quan, **không nhập Accuracy vào bảng isolated v1** |
| [Zhang et al. BiLSTM–Transformer, Electronics](https://www.mdpi.com/2079-9292/15/12/2578) | 2026 | Traffic police, pose→sequence | Nghiên cứu liên quan, khác 11-video isolated RGB |
| [IPN HandS, Applied Sciences](https://www.mdpi.com/2076-3417/15/11/6321) | 2025 | IPN refined annotation + skeleton | Biến thể nhãn, không tráo với IPN original |
| [Multimodal RGB-derived optical flow/depth on IPN, SoMeT](https://doi.org/10.3233/FAIA240397) | 2024 | IPN nhưng chỉnh annotation và số lớp | Related work; phải kiểm định nhãn/phiên bản trước khi so |
| [Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks](https://iris.uniroma1.it/handle/11573/1735312) | 2025 | IPN landmark-based, subject-wise cross-validation | Không cùng raw RGB & official held-out evaluation |
| [Xiong et al., Electronics](https://www.mdpi.com/2079-9292/10/5/551) | 2021 | CTPGesture v1 skeleton; bảng RGB 3D CNN | **Baseline lịch sử** bảng 3, không quảng bá là SOTA mới |
| [IPN-Hand original benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html) | 2020–21 | RGB/Flow/Seg, isolated và Levenshtein | **Baseline lịch sử** bảng 1–2 |

**Kết quả khảo sát:** Với chứng cứ hiện có, chưa xác minh được `~5 bài RGB-only, CTPGesture v1, 2023–2026, cùng test split và isolated/continuous protocol` để điền số liệu trực tiếp. **Không tự điền 5 hàng không cùng giao thức** để đủ số lượng. Nếu cần bảng fair RGB baselines, tái triển khai ít nhất một mô hình như C3D, R3D-18/ResNet3D hoặc VideoMAE-Small với **cùng manifest và evaluation code**.

## H. CHECKLIST ƯU TIÊN — TẬN DỤNG OUTPUT, KHÔNG CHẠY LẠI GPU

### P0 — Làm ngay, không cần GPU

- [x] Lập 4 bảng cho 2 dataset × (isolated, continuous), có liên kết tới nguồn từng công trình.
- [x] Gắn trạng thái đối chứng A/B/C và tách RGB, flow-derived, skeleton, causal/offline.
- [x] Kiểm tra lại công thức Mean per-video Frame Accuracy của CoboGesture; ghi đúng **83.4889% Raw**, **83.5249% R3** cho IPN.
- [x] Kiểm tra notebook CTP lưu split test, Macro-F1 và output Raw; ghi rõ protocol hiện tại.
- [x] Đánh dấu R3 là `test-tuned`, không dùng kết luận vượt SOTA.
- [ ] Kiểm tra danh sách **1.101 IPN isolated samples** và 52 video continuous so với split CoboGesture (nếu xác minh được danh sách từ tác giả).
- [ ] Gom và giữ bản sao checkpoint, per-video prediction CSV, raw probability cache, annotation manifest, môi trường và commit SHA cho cả hai dataset (các tệp `/kaggle/working` chưa được xác nhận còn tồn tại sau phiên).
- [ ] Ghi một bảng phiên bản `dataset / split / frames / checkpoint / code commit / input sampling` cố định.

### P1 — CPU hoặc xử lý từ cache nếu tệp còn

- [ ] Unit test cho frame aggregation, Temporal IoU/Overlap, unmatched predictions, boundary `+1`, videos rỗng và metric mean-vs-global.
- [ ] Tính Levenshtein Accuracy IPN từ chuỗi nhãn đã lưu; chỉ chạy nếu còn per-frame predictions, không suy từ bảng tổng hợp.
- [ ] Tạo validation raw cache (nếu còn) và chọn hậu xử lý chỉ trên validation; khóa config trước test.
- [ ] Đối chiếu cách đếm MACs/FLOPs với profiler, liệt kê unsupported Mamba ops; nhất quán `GFLOPs/clip` và `GFLOPs/12-view sample`.
- [x] Bổ sung Params vào 4 bảng từ IPN gốc, CoboGesture, RSC code; phân biệt classifier và full pipeline.
- [ ] Cần xác minh Params toàn pipeline của RSC (pose estimator/descriptor CNN), nguồn checkpoint baseline còn thiếu và variant CTP lịch sử; để `—` khi không xác thực.
- [ ] Lập bản biểu `mean±std / CI` bằng bootstrap theo video từ output đã lưu, không báo cáo chỉ một con số phần trăm.

### P2 — Chỉ thực hiện khi đã khóa giao thức lần chạy tiếp theo

- [ ] CTP: tạo **causal end-of-window** evaluation, không dùng frame tương lai; tính Online-F1, TTC, Early@10 theo đúng định nghĩa trong RSC-GestureNet.
- [ ] IPN: nếu thật sự cần fair held-out R3, tách validation ở cấp video/subject theo quy định dataset, chọn ngưỡng trên validation.
- [ ] Nếu chưa đủ đối chứng RGB: chạy lại **một hoặc hai** baseline trên cùng dataset manifest/protocol thay vì huấn luyện hàng loạt.
- [ ] Một lần chạy GPU tiếp theo phải lưu `window_logits / timestamps / per-frame predictions / ground_truth / metadata` để các metric đánh giá lại được **không cần forward lại**.
- [ ] EgoEvGesture là nhánh tiếp theo, chưa đưa vào 4 bảng hiện tại khi chưa có metric kiểm thử.

## I. Kết luận dùng khi gửi thầy

**Đã hoàn thiện:** bốn bảng tài liệu và đối chiếu bước đầu; metric và số liệu nội bộ được truy ngược tới notebook; nghiên cứu liên quan 2023–2026 được phân loại theo giao thức. Kết quả Raw của VideoMamba có thể dùng **làm baseline hiện có**: IPN isolated **84.65%**, IPN continuous **83.4889% mean-per-video**, CTP isolated **98.5532%**, CTP continuous **83.9836% global**/**80.6135% Macro-F1**.

**Chưa thể khẳng định:** VideoMamba vượt SOTA về hiệu năng nhận dạng liên tục, vì thiếu giao thức causal tương đương RSC, R3 được dò trên test, và không phải mọi bài báo sử dụng cùng split/metric. Cũng chưa thể kết luận GFLOPs là chuẩn hóa xuyên framework.

**Đề nghị thứ tự công việc:** (1) khóa và sao lưu output; (2) xác minh source & protocol 4 bảng; (3) chuẩn hóa metric offline bằng CPU/cache; (4) mới quyết định thí nghiệm GPU tối thiểu; (5) sau đó mới chuyển EgoEvGesture. Đây là phương án tránh tiêu tốn GPU cho kết quả mà bảng đối chứng chưa cần.

## J. AUDIT PARAMETER COUNTS (bổ sung 27/10/2026)

| Phạm vi / đối tượng | Trainable parameters | Bằng chứng | Kết luận sử dụng |
|---|---:|---|---|
| VideoMamba IPN-Hand, 14 lớp | 6.9608M | IPN notebook, `sum(p.numel())` | Toàn bộ RGB recognition network |
| VideoMamba CTPGesture, 9 lớp | 6.959817M | CTP notebook, `sum(p.numel())` | Toàn bộ RGB recognition network |
| CoboGesture VideoMAEv2 ViT-L | 303.8M | Bài IEEE Access 2026 Table 4 | Approx classifier |
| IPN ResNeXt-101 RGB | 47.51M | IPN gốc Table IV | Classifier |
| IPN ResLight-10 RGB | 0.895M | IPN gốc Table VIII | Detector cần cộng vào hierarchical continuous |
| IPN HarDNet / SPyNet | 4.114M / 1.440M | IPN gốc Table V | Extra learned preprocessing nếu RGB-Seg/Flow |
| RSC fusion, `RSCGestureNet` | **815,270 = 0.815270M** | `src/rsc_gesturenet/model.py`, `configs/train.yaml` | Recognition-only, không gồm pose detector |
| RSC pose backbone, `RSCGestureBackbone` | **335,531 = 0.335531M** | cùng nguồn | Chưa kiểm tra mapping đầy đủ với checkpoint Pose+VC |
| RSC MDGCN reimplementation | **399,993 = 0.399993M** | `src/rsc_gesturenet/baselines.py`, `configs/train.yaml` | **Không phải** số của MD-GCN 2021 |
| RSC HLP-GCN-style reimplementation | **335,817 = 0.335817M** | cùng nguồn | Không điền vào bảng 4 khi chưa ghép với metric paper |
| Zhang 2026 recognition head | **86.41K = 0.08641M** | Electronics 2026 Table 3 | Chỉ head |
| YOLO11m-Pose | khoảng **20.9M** | Model catalog Ultralytics | Cộng vào Zhang head để ước tính **≈20.99M** full pipeline |

**Cách xác minh code-derived RSC bằng CPU (không inference):** khi clone repo RSC và cài PyTorch trên máy bất kỳ, tạo `RSCGestureNet(num_joints=17,num_classes=9,hidden=96,dropout=0.12,rgb_dim=12)`, `RSCGestureBackbone(17,9,96,0.12)`, `MDGCN(17,9,96,0.12)`, rồi `sum(p.numel() for p in model.parameters())`. Đếm cấu trúc theo đúng layer và kích thước đó cho ra **815270 / 335531 / 399993**. Khối `GRU`, `TCN`, `LayerNorm`, các bias và head đều được tính. Khi dùng một checkpoint cụ thể cần **khớp architecture và version** của code tại thời điểm tác giả release.

**Không tự đồng nhất giữa:** (1) model file size MB, (2) số phần tử tensor trong `state_dict` (có buffer), (3) trainable parameters, (4) FLOPs/MACs, (5) tham số full end-to-end pipeline từ RGB. Chỉ dùng cùng định nghĩa Params để rút ra tỷ lệ model size.

## K. TRUY VẾT CITATION ĐỂ HOÀN THIỆN THÔNG SỐ — 27/10/2026

**Cách thực hiện:** paper đối chứng → reference paper → nguồn tác giả/model zoo/checkpoint → xác định kiến trúc và input đúng biến thể → tham số hoặc FLOPs. Chia bốn mức chứng cứ: **[P]** trực tiếp đúng model trong bài; **[C]** đếm từ code với config đúng; **[R]** số của kiến trúc cùng họ nhưng **chưa chứng minh trùng checkpoint benchmark**; **[E]** số suy ra hoặc ước lượng theo giả thiết. Không dùng [R]/[E] để tuyên bố thắng SOTA.

### K.1. Bảng bổ sung đã truy ngược đến nguồn gốc

| Mô hình và đúng biến thể | Parameters (M) | GFLOPs/clip | Input / giới hạn | Chứng cứ |
|---|---:|---:|---|---|
| VideoMamba-Tiny nhóm IPN / CTP | 6.9608 / 6.959817 | 29.0261 | 16×224², FLOPs đếm thủ công của nhóm | [C]/[E] |
| VideoMamba-Tiny original | khoảng 7 | khoảng 17 | 16×224², FLOPs theo định nghĩa công trình ECCV 2024 | [R] |
| VideoMAE v1/v2 ViT-L/16 do nhóm dựng | 303.8751 | 596.8327 | 16×224², 14-class, định nghĩa 1 MAC = 1 FLOP-equivalent | [C]/[E] |
| C3D IPN-Hand original | 50.75 | — | RGB 32 frames, bảng IPN gốc | [P] |
| C3D MMAction2 | 78.4 | 38.5 | 16×112², UCF-101, **khác C3D IPN/CTP** | [R] |
| R3D-18 Torchvision Kinetics400 | 33.371472 | 40.697 | 16×112², **chưa xác minh** là bản Xiong sử dụng | [R] |
| RSC-GestureNet Fusion recognition | 0.815270 | — | model.py, hidden=96, 17 joints, rgb_dim=12; **chưa gồm pose estimator** | [C] |
| RSC pose backbone | 0.335531 | — | Cùng code/config; chưa gồm bộ trích pose | [C] |
| MD-GCN-style RSC reimplementation | 0.399993 | — | baselines.py, khác Xiong 2021 | [C] |
| HLP-GCN-style RSC reimplementation | 0.335817 | — | baselines.py; đối chiếu checkpoint theo kiến trúc | [C] |
| ST-GCN của CoboGesture Table 3 | 2.8 | — | Trên dataset CoboGesture; **không chứng minh** ST-GCN của RSC dùng cùng cấu hình | [P] trong Cobo / [R] ngoài Cobo |
| YOLO11m-Pose + BiLSTM–Transformer | khoảng 20.99 | — | 20.9 detector + 0.08641 recognition head, tổng suy ra | [E] |
| FDMB-YOLOv11 2026 | 1.93 | — | **Single-frame RGB detection**, mAP@0.5 = 97.17%, không phải Accuracy video | [P], related work |

**Nguồn và phạm vi:** [IPN-Hand ICPR Table IV, V, VIII, IX](https://gibranbenitez.github.io/2021_ICPR_IpnHand.pdf); [official IPN leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html); [Torchvision R3D-18 model zoo](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html); [MMAction2 C3D model zoo](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md); [VideoMamba ECCV 2024, Kunchang Li et al.](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3773_ECCV_2024_paper.php); [CoboGesture 2026](https://doi.org/10.1109/ACCESS.2026.3664027); [RSC-GestureNet model.py](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py), [baselines.py](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/baselines.py), [train.yaml](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/configs/train.yaml); [PLOS ONE FDMB-YOLOv11, 28/08/2026](https://doi.org/10.1371/journal.pone.0357212).

### K.2. Những ô trống không thể suy luận như số exact

1. **MD-GCN Xiong 2021:** Model Size = 11.7 M không rõ đơn vị. Trường hợp giả thiết 11.7 MB FP32 weights-only suy ra khoảng 2.925M, nhưng không có đủ bằng chứng dùng làm Params công bố.
2. **3D ResNet/C3D/P3D của Xiong:** Bảng paper chỉ liệt kê tên họ mô hình và Accuracy; model zoo khác có thể khác độ sâu, số lớp head, input, nên các số 33.371472/78.4 chỉ là kiến trúc [R].
3. **RSC đầy đủ:** Recognition fusion 0.815270M nhưng model nhận keypoints và RGB descriptors đã xử lý. Chưa có tổng Params của chính xác MediaPipe estimator và bất kỳ mạng tạo descriptor có learned weights.
4. **GRU, ST-GCN, TCN+VC của RSC:** Chưa tìm thấy class/checkpoint/config đủ xác minh đúng các hàng Table 1. ST-GCN = 2.8M trong Cobo không phải bằng chứng cho ST-GCN reimplemented RSC.
5. **GFLOPs giữa phương pháp:** 17G vs 29.0261G VideoMamba là **hai quy ước profiling** khác nhau. 596.8327G của VideoMAE thuộc phép đo của nhóm; 38.5/40.697G của model zoo là 16×112², không thể coi trực tiếp so với 16×224².
6. **Bài mới FDMB-YOLOv11:** Có tham số 1.93M nhưng là nhận dạng ảnh tĩnh với mAP, không nên thêm vào Bảng CTP isolated/continuous với Accuracy giả định.

**Kết luận:** Citation chaining đã giúp bổ sung thông số có nguồn cho họ 3D CNN/VideoMAE/VideoMamba và nhận dạng skeleton 2026, không tốn GPU. Để bảng có giá trị khoa học, nên giữ cột nguồn và mã [P]/[C]/[R]/[E]; không điền các con số chính xác giả tạo cho biến thể không thể xác minh.
