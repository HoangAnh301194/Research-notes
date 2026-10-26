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

### Bảng 1 — IPN-Hand: Isolated Gesture Recognition

**Cột ưu tiên: Parameters (M).** Với mô hình RGB đơn, số tham số của classifier chính là toàn bộ recognition network. Với RGB+Flow/Seg, cần cộng thêm **mô hình sinh modality** khi đánh giá tổng pipeline. Notebook nhóm đánh giá 1.101 đoạn cử chỉ, loại class non-gesture, input 16×224×224; các baseline IPN gốc chủ yếu dùng 32 frames.

| Method / nguồn | Năm | Input | **Params classifier (M)** | **Params pipeline (M)** | GFLOPs/clip | Accuracy (%) | Ghi chú |
|---|---:|---|---:|---:|---:|---:|---|
| **VideoMamba-Tiny [O]** | 2026 | RGB, 16f | **6.9608** | **6.9608** | **29.0261 [E]** | **84.65** | 1.101 clips; kiểm tra giao thức |
| RGB-CoGes + VideoMAEv2 [P] | 2026 | RGB | **303.8** | **303.8** | — | 84.59 | [CoboGesture Table 4] |
| RGB-CoGes + VideoMAE [P] | 2026 | RGB | **303.8** | **303.8** | — | 78.81 | [CoboGesture Table 4] |
| ResNeXt-101 [P] | 2020/21 | RGB + Flow | **47.56** | **49.000*** | — | 86.32 | SPyNet 1.440M |
| ResNeXt-101 [P] | 2020/21 | RGB + Seg | **47.56** | **51.674*** | — | 84.77 | HarDNet 4.114M |
| ResNeXt-101 [P] | 2020/21 | RGB | **47.51** | **47.51** | — | 83.59 | 32f |
| C3D [P] | 2020/21 | RGB | **50.75** | **50.75** | — | 77.75 | 32f, biến thể IPN gốc |
| ResNet-50 [P] | 2020/21 | RGB | **46.25** | **46.25** | — | 73.10 | 32f |

**Nguồn Params (bản gốc):** [Benitez-Garcia et al., IPN-Hand, Table IV/V](https://arxiv.org/abs/2005.02134) ([toàn văn có bảng](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition)); [Nguyen et al., CoboGesture 2026, Table 4](https://doi.org/10.1109/ACCESS.2026.3664027); [notebook VideoMamba](../261007/videomamba_man_configsBenchmark.ipynb). `*` Tổng suy ra bằng cách cộng các mô hình **trong chuỗi suy luận**: 47.56+1.440=49.000M (Flow), 47.56+4.114=51.674M (Seg); không phải tổng được tác giả in sẵn. Nếu tiền xử lý Flow/Seg chỉ chạy offline và cache, chi phí inference thực tế cần báo riêng.

**Ý nghĩa:** VideoMamba ít tham số hơn ViT-L của CoboGesture khoảng **303.8/6.9608 = 43.64 lần** trong định nghĩa recognition network, nhưng chênh lệch isolated accuracy **0.06 điểm phần trăm** không đủ chứng minh vượt SOTA. *Parameters ≠ FLOPs ≠ latency.*

### Bảng 2 — IPN-Hand: Continuous Gesture Recognition

**Bổ sung Params đặc biệt quan trọng:** Continuous pipeline của benchmark IPN gốc có **detector ResLight-10 + classifier**; nếu dùng RGB-Flow/Seg phải cộng thêm mạng SPyNet/HarDNet. **Không** so `47.51M` của classifier-only với `6.96M` của model end-to-end mà không nêu phạm vi. RGB-CoGes dùng lại VideoMAE/VideoMAEv2 và hậu xử lý không học trọng số, nên vẫn khoảng 303.8M learned parameters.

| Method / nguồn | Input | **Params recognition (M)** | **Params toàn pipeline (M)** | Mean per-video FrameAcc (%) | Activity overlap | Levenshtein (%) | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---|
| **VideoMamba-Tiny Raw [O]** | RGB | **6.9608** | **6.9608** | **83.4889** | 0.491110 | — | Offline window aggregation |
| VideoMamba-Tiny R3 [O] | RGB | 6.9608 | 6.9608 | 83.5249 | 0.507705 | — | **Test-tuned** |
| RGB-CoGes + VideoMAEv2 [P] | RGB | **303.8** | **303.8** | 83.00 | 0.510 | — | Cobo Table 7 |
| RGB-CoGes + VideoMAE [P] | RGB | **303.8** | **303.8** | 80.00 | 0.370 | — | Cobo Table 7 |
| DD-Net continuous [P] | Skeleton | **2.4†** | **—** | 57.00 | — | — | 2.4M chỉ từ **DD-Net architecture ở Cobo Table 3**, chưa xác nhận bản IPN; pose extractor chưa gồm |
| ResNeXt-101 + ResLight + SPyNet [P] | RGB + Flow | 47.56 | **49.908*** | — | — | 42.47 | Classification 47.56 + detector 0.908 + flow 1.440 |
| ResNeXt-101 + ResLight [P] | RGB | 47.51 | **48.405*** | — | — | 25.34 | 47.51 + 0.895 detector |
| ResNet-50 + ResLight + SPyNet [P] | RGB + Flow | 46.27 | **48.618*** | — | — | 39.47 | 46.27 + 0.908 + 1.440 |
| ResNet-50 + ResLight [P] | RGB | 46.25 | **47.145*** | — | — | 19.78 | 46.25 + 0.895 |

**Nguồn:** [IPN-Hand gốc, Tables IV, V, VIII và IX](https://www.researchgate.net/publication/341175756_IPN_Hand_A_Video_Dataset_and_Benchmark_for_Real-Time_Continuous_Hand_Gesture_Recognition); [CoboGesture 2026, Tables 3/4/7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction); [IPN notebook](../261007/videomamba_man_configsBenchmark.ipynb).

`*` Tổng **tính từ số liệu Parameters của các mạng con**, giả thiết các mạng này đều được triển khai ở thời điểm suy luận. Không lấy model size tính bằng MB làm số parameters. `†` Giá trị kiến trúc tham chiếu, **không chứng minh** pipeline DD-Net IPN hiện có đúng 2.4M.

**Định nghĩa metric:** CoboGesture tính Frame-wise Accuracy bằng trung bình theo video. Do đó ưu tiên Raw `83.4889%` của nhóm so với `83.00%` của Cobo VideoMAEv2. Global FrameAcc Raw `83.7308%` và R3 `83.7671%` không được thay thế cho Mean FrameAcc. Các số Levenshtein **khác metric**, không xếp cùng thứ hạng. R3 dùng ngưỡng chọn trên test, chỉ để phân tích, không xem là test độc lập.

### Bảng 3 — CTPGesture v1: Isolated Gesture Recognition

**Yêu cầu Parameters:** Tách số tham số **thực sự được bài báo công bố** khỏi tham số của một kiến trúc **tham khảo khác**. Đặc biệt, model file size (MB) **không** đồng nghĩa với số lượng parameters (M).

| Method / nguồn | Năm | Input | **Params classifier / recognition (M)** | **Params toàn pipeline (M)** | GFLOPs/clip | Accuracy (%) | Macro-F1 (%) | Đối chiếu |
|---|---:|---|---:|---:|---:|---:|---:|---|
| **VideoMamba-Tiny, 9 lớp [O]** | 2026 | RGB | **6.959817** | **6.959817** | 29.0261 [E] | **98.5532** | 97.9590 | Official 11 test videos, 1.728 segments |
| VideoMamba, gesture-only [O] | 2026 | RGB | 6.959817 | 6.959817 | 29.0261 [E] | 98.0210 | 98.2363 | **Khác subset**, loại class-0 GT |
| GSE + MD-GCN (Xiong et al.) [P] | 2021 | OpenPose + skeleton | **—** | **—** | — | 98.95 | — | Paper Table 1 ghi **model size 11.7 M**, **không phải** params 11.7M; khác test split |
| P3D (Qiu, baseline Xiong) [P] | 2021 | RGB | **≈65.3†** | **—** | — | 92.21 | — | † Ước từ **P3D-199 tham khảo**, không biết chính xác variant Xiong |
| C3D (Tran, baseline Xiong) [P] | 2021 | RGB | **78.4†** | **—** | — | 87.12 | — | † MMAction2 C3D 16×112²; **không** gán cho Xiong như số đo |
| 3D ResNet (Hara, baseline Xiong) [P] | 2021 | RGB | **33.4†** | **—** | — | 81.16 | — | † torchvision R3D-18 reference, không xác minh Xiong variant |
| YOLO11m-Pose + BiLSTM–Transformer (Zhang et al.) [P] | 2026 | RGB→Pose | **0.08641 [P, recognition head]** | **≈20.98641‡** | — | 98.91 | — | **Nghiên cứu liên quan; sequence-to-one, 10 video**, không phải isolated protocol nhóm |
| EPFFNet + MSNet (NTPGR, Xiao et al.) [P] | 2025 | RGB→Pose | **—** | **—** | — | 97.56 | — | **Gesture-instance accuracy**, khác protocol/dataset partition |

**Nguồn:**
- [Xiong et al. 2021, Tables 1 và 3 — kích thước MD-GCN 11.7 M và Accuracy 98.95%](https://www.mdpi.com/2079-9292/10/5/551).
- [Zhang et al. 2026, Table 3 — recognition head 86.41 K](https://www.mdpi.com/2079-9292/15/12/2578), [Ultralytics YOLO11m-pose — detector ≈20.9M](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md).
- [C3D MMAction2 — 78.4M cho một biến thể cụ thể](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md), [Torchvision R3D-18 — 33.4M cho một biến thể cụ thể](https://docs.pytorch.org/vision/master/models.html), [P3D original (Qiu 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf), [NTPGR 2025](https://www.nature.com/articles/s41598-025-02833-y).
- [CTPGesture notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb).

`†` **Architecture-reference-only:** số chỉ mô tả biến thể phổ biến, chưa xác minh chính xác mạng mà Xiong dùng. Do đó **không thể dùng để kết luận tỷ lệ nén hay độ phức tạp relative to Xiong**. P3D-199 ≈65.3M suy từ 261MB FP32 (xấp xỉ 65.25 triệu tham số, giả thiết kích thước chỉ là tensor weights). `‡` ≈20.9M + 0.08641M = **≈20.99M**, là **ước tính pipeline Zhang** dùng detector chính thức và classifier head của bài báo, không phải số total được bài báo đo trực tiếp.

**Đặc biệt:** **11.7 M** trong Xiong là cột *Model Size*, không phải cột *No. Parameters*. Không được đặt `11.7` vào cột Params (M) dù chữ M giống nhau. Cần source checkpoint/implementation để đếm chính xác, đồng thời tính cả OpenPose nếu so full pipeline.

### Bảng 4 — CTPGesture v1: Continuous Gesture Recognition

**Cột Params đo từ code gốc RSC:** Công khai [model.py](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py), [baselines.py](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/baselines.py), [train.yaml](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/configs/train.yaml). Với `hidden=96`, `num_joints=17`, `num_classes=9`, `rgb_dim=12`, **đếm tất cả trainable parameters bằng kiến trúc PyTorch tái tạo trên CPU (không chạy inference):** fusion = **815.270**, pose backbone = **335.531**, MD-GCN-style reimplementation = **399.993**. **Các số dưới đây chưa gồm RGB→pose detector.**

| Method / nguồn | Input / protocol | **Params recognition (M)** | **Params RGB→pose (M)** | **Params toàn pipeline (M)** | Accuracy (%) | Macro-F1 (%) | Online-F1 (%) | Early@10 (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **VideoMamba Raw [O]** | RGB, offline, 32/8 | **6.959817** | Không dùng | **6.959817** | **83.9836** | **80.6135** | — | — |
| VideoMamba R3 [O, test-tuned] | RGB offline + hậu xử lý | 6.959817 | Không dùng | 6.959817 | 83.9940 | 80.6254 | — | — |
| RSC-GestureNet Fusion [C, code gốc 2026] | Pose + RGB descriptor, causal | **0.815270** | **—** | **—** | 93.33±0.24 | 91.71±0.27 | 91.69±0.29 | 98.80±0.07 |
| RSC-Gesture-Pose+VC [E, kiến trúc pose] | Pose, causal | **≈0.335531†** | **—** | **—** | 93.09±0.26 | 91.35±0.43 | 91.33±0.39 | 98.41±0.34 |
| TCN+VC [P] | Pose/temporal features, causal | **—** | **—** | **—** | 92.97±0.18 | 91.16±0.21 | 91.08±0.23 | 98.41±0.07 |
| MD-GCN-style reimplemented [C, RSC source] | Pose, causal | **0.399993** | **—** | **—** | 90.08±1.38 | 88.48±1.43 | 89.54±0.76 | 98.14±0.42 |
| ST-GCN [P, RSC baseline] | Pose, causal | **—** | **—** | **—** | 89.45±0.65 | 87.94±0.72 | 82.64±0.43 | 98.06±0.41 |
| GRU [P, RSC baseline] | Pose/temporal, causal | **—** | **—** | **—** | 88.82±0.43 | 87.10±0.45 | 81.99±0.82 | 98.14±0.31 |

**Nguồn:** [Li, Gao & Fu, RSC-GestureNet (2026), kết quả thực nghiệm](https://arxiv.org/abs/2608.02200), [repo tác giả](https://github.com/chengli24/rsc-gesturenet-prcv2026), [CTP notebook của nhóm](../261015/VideoMambaTinyCTPGestureFinal.ipynb). `[C]` = **code-derived count**, `[P]` = paper-reported metric, `[E]` = ước tính/cần khớp checkpoint. `†` `RSCGestureBackbone` có 0.335531M tham số; cần xác nhận biến thể Pose+VC trong bảng paper đúng checkpoint/architecture này trước khi coi là exact. `0.399993M` thuộc **bản MD-GCN-style do nhóm RSC tự tái triển khai**, **không** phải tham số của Xiong et al. 2021.

**Vấn đề tiên quyết:** RSC sử dụng đặc trưng keypoints/pose được trích từ RGB. Nếu nhận bộ keypoints tiền xử lý sẵn thì mô hình nhận dạng chỉ 0.815M; nếu so *RGB-to-command end-to-end* phải cộng **mô hình pose estimator** và bất kỳ CNN tạo RGB descriptors có học trọng số. Chưa xác minh trọn bộ extractor từ bản công bố, vì vậy **không tự cộng một BlazePose 1.3M không chắc đã được sử dụng**. Hướng nghiên cứu của RSC **nhẹ hơn ở recognition head**, nhưng không thể dùng cột 0.815M để tuyên bố pipeline đó nhẹ hơn VideoMamba RGB 6.96M khi không tính detector.

**Đánh giá:** VideoMamba sử dụng probability averaging trên toàn window (phi nhân quả); RSC là causal end-of-window; do đó **không xếp trực tiếp 83.98% vs 93.33%**. Các giá trị Temporal Overlap Raw `0.653223`, R3 `0.658979` thuộc metric khác Online-F1 và Early@10. RSC Early@10 chỉ khả năng nhận đúng trong 10 bước đầu ra sau chuyển lệnh, **không phải Accuracy@10% số frame đầu gesture**.


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
