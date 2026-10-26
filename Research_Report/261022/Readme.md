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

**Chỉ số chính:** Top-1 Accuracy trên các clip cử chỉ với ground-truth boundaries. Notebook VideoMamba dùng 1.101 mẫu test, loại non-gesture `D0X`; đầu vào RGB 16×224×224. Các giá trị từ CoboGesture lấy ở **Table 4**; baseline đầu tiên của IPN-Hand lấy từ benchmark chính thức.

| Method / nguồn | Năm | Input | Params (M) | GFLOPs/clip | Accuracy (%) | Mức đối chiếu |
|---|---:|---|---:|---:|---:|---|
| **VideoMamba-Tiny [O]** | 2026 | RGB, 16 frames | **6.9608** | **29.0261 [E]** | **84.65** | A — cần kiểm tra trùng 1.101 mẫu |
| RGB-CoGes + VideoMAEv2 [P] | 2026 | RGB | 303.8 [P] | — | 84.59 | A — Table 4 |
| RGB-CoGes + VideoMAE [P] | 2026 | RGB | 303.8 [P] | — | 78.81 | A — Table 4 |
| ResNeXt-101 [P] | 2020/21 | RGB + optical flow | — | — | 86.32 | B — đa đầu vào, 32 frames |
| ResNeXt-101 [P] | 2020/21 | RGB + segmentation | — | — | 84.77 | B — đa đầu vào, 32 frames |
| ResNeXt-101 [P] | 2020/21 | RGB | — | — | 83.59 | B — 32 frames |
| C3D [P] | 2020/21 | RGB | 50.75 [P, Cobo] | — | 77.75 | B |
| ResNet-50 [P] | 2020/21 | RGB | 46.25 [P, Cobo] | — | 73.10 | B |

Nguồn: [CoboGesture (Nguyen et al., IEEE Access 2026), Table 4](https://doi.org/10.1109/ACCESS.2026.3664027), [toàn văn CoboGesture trên ResearchGate](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction), [benchmark chính thức IPN-Hand](https://gibranbenitez.github.io/IPN_Hand/Results.html), [GitHub gốc IPN-Hand](https://github.com/GibranBenitez/IPN-hand). 29.0261 GFLOPs là **bộ đếm thủ công** của notebook nhóm; giá trị **596.8327 GFLOPs** của VideoMAE ViT-L/16 ở notebook nhóm chỉ là benchmark kiến trúc dùng cấu hình cục bộ, **không** gán cho Table 4 của CoboGesture.

**Không kết luận 84.65% > 84.59% về mặt thống kê:** độ chênh chỉ 0.06 điểm phần trăm, khác thiết lập và chưa có khoảng tin cậy. Chỉ số 86.32% dùng optical flow (thêm modality), không phải kết quả RGB-only.

### Bảng 2 — IPN-Hand: Continuous Gesture Recognition

**Giao thức:** VideoMamba có 52 video test, cửa sổ 64 frame/step 16 ở 30 FPS, sample 16 frame. CoboGesture **Eq. (4)–(6)** định nghĩa Frame-wise Accuracy = trung bình Accuracy của **từng video**, vì vậy dùng `Mean per-video FrameAcc` trong bảng chính; `Global FrameAcc` không thay thế được. CoboGesture Table 7 nêu activity overlap score.

| Method / nguồn | Input | Mean per-video FrameAcc (%) | Activity overlap | Gesture Recall (%) | Levenshtein Accuracy (%) | Mức đối chiếu |
|---|---|---:|---:|---:|---:|---|
| **VideoMamba-Tiny Raw [O]** | RGB | **83.4889** | **0.491110** | 78.8995 | — | A — chuẩn hóa metric |
| VideoMamba-Tiny R3 [O, test-tuned] | RGB | 83.5249 | 0.507705 | 76.2844 | — | **B — không phải test độc lập** |
| RGB-CoGes + VideoMAEv2 [P] | RGB | 83.00 | 0.510 | — | — | A — Table 7, cần audit overlap |
| RGB-CoGes + VideoMAE [P] | RGB | 80.00 | 0.370 | — | — | A — Table 7 |
| DD-Net continuous [P, báo cáo lại trong Cobo] | Skeleton | 57.00 | — | — | — | B — khác input |
| ResNeXt-101 IPN official [P] | RGB + optical flow | — | — | — | 42.47 | **C — khác metric** |
| ResNet-50 IPN official [P] | RGB + optical flow | — | — | — | 39.47 | **C — khác metric** |
| ResNeXt-101 IPN official [P] | RGB | — | — | — | 25.34 | **C — khác metric** |

Nguồn: [CoboGesture 2026, Eq. 4–7, Table 7](https://www.researchgate.net/publication/400740972_CoboGesture_A_continuous_hand_gesture_dataset_and_recognition_method_for_human-collaborative_robot_interaction), [IPN-Hand official continuous benchmark](https://gibranbenitez.github.io/IPN_Hand/Results.html), [notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb).

**Số không được thay thế lẫn nhau:** VideoMamba Raw `Global FrameAcc=83.7308%`; R3 `Global FrameAcc=83.7671%`. Đây là Accuracy có trọng số theo tổng frame, **không** phải phép lấy trung bình video mà CoboGesture công bố. Độ chênh danh nghĩa giữa R3 và CoboGesture VideoMAEv2 ở **cột Mean FrameAcc** = 83.5249−83.00 ≈ **+0.52 điểm phần trăm**, **không** phải +0.77. R3 được chọn ngưỡng trên test, vì vậy ưu tiên so sánh Raw hoặc tái đánh giá R3 với ngưỡng chọn trên validation. Code Temporal Overlap dùng IoU, matching cùng nhãn, tolerance 2 s; cần test thêm xử lý unmatched, endpoint inclusivity (`+1` frame), và per-video aggregation trước khi khẳng định đúng hệt Eq. (7) của CoboGesture. Levenshtein Accuracy (IPN official) là metric **khác**, chưa thể đưa vào cùng thứ hạng.

### Bảng 3 — CTPGesture v1: Isolated Gesture Recognition

**Giao thức của nhóm:** 11 video test, 1.728 contiguous segments (869 non-gesture + 859 gesture), dùng ground-truth boundaries và uniform sampling 16 RGB frames/segment.

| Method / nguồn | Năm | Input | Params toàn pipeline (M) | GFLOPs/clip | Accuracy (%) | Macro-F1 (%) | Mức đối chiếu |
|---|---:|---|---:|---:|---:|---:|---|
| **VideoMamba-Tiny, 9 lớp [O]** | 2026 | RGB | **6.959817** | **29.0261 [E]** | **98.5532** | **97.9590** | A — nhóm |
| VideoMamba-Tiny, gesture-only [O] | 2026 | RGB | 6.959817 | 29.0261 [E] | 98.0210 | 98.2363 | **B — bỏ GT class 0**, không dùng xếp hạng 9 lớp |
| GSE + MD-GCN (Xiong et al.) [P] | 2021 | Skeleton/GCN | — | — | 98.95 | — | **B — 9 test videos/1.565 clips** |
| P3D, đối chứng Xiong [P] | 2021 | RGB | — | — | 92.21 | — | B — split khác |
| C3D, đối chứng Xiong [P] | 2021 | RGB | — | — | 87.12 | — | B — split khác |
| 3D ResNet, đối chứng Xiong [P] | 2021 | RGB | — | — | 81.16 | — | B — split khác |

Nguồn: [Xiong et al., *Electronics* 2021, Table 3](https://www.mdpi.com/2079-9292/10/5/551), [PDF nghiên cứu đã lưu trong repo](../261015/electronics-10-00551-v2.pdf), [CTP final notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb).

**Loại khỏi bảng so sánh isolated trực tiếp (nhưng tiếp tục nghiên cứu):**
- [Zhang et al., BiLSTM–Transformer, *Electronics* 2026](https://www.mdpi.com/2079-9292/15/12/2578): 98.91% reported ở protocol sequence-to-one/skeleton (YOLO pose), **không** phải cùng 11-video, 1.728-segment isolated RGB benchmark. Hệ thống có 10 test videos theo báo cáo 261015; không gộp vào thứ hạng isolated.
- [Xiao, Li & Zhao, NTPGR, *Scientific Reports* 2025](https://www.nature.com/articles/s41598-025-02833-y): 97.56% theo **gesture instances** trên cấu trúc dataset/giao thức khác; không dùng như 97.56% clip accuracy tương đương.
- Công trình về [IPN HandS 2025](https://www.mdpi.com/2076-3417/15/11/6321) là **biến thể annotation/skeleton**, không dùng số của HandS như số IPN-Hand RGB nguyên gốc.

**Lưu ý:** 98.95% của MD-GCN không chứng minh trực tiếp tốt hơn hoặc kém VideoMamba 98.55% vì khác test subset, input và protocol. Không chèn số params/GFLOPs suy ra từ P3D/C3D/R3D biến thể khác vào bảng chính.

### Bảng 4 — CTPGesture v1: Continuous Gesture Recognition

**Hai nhóm không cùng giao thức.** VideoMamba gán nhãn từng frame bằng trung bình xác suất **tất cả cửa sổ chứa frame**, trong đó cửa sổ có thể chứa frame tương lai (phi nhân quả, offline). RSC-GestureNet đánh giá bằng **causal full-frame windows** stride 4 và cung cấp Online-F1/TTC/Early@10. **Cùng tên dataset không đồng nghĩa cùng protocol.**

| Method / nguồn | Input / protocol | Accuracy (%) | Macro-F1 (%) | Online-F1 (%) | Early@10 (%) | TTC (s) | Mức đối chiếu |
|---|---|---:|---:|---:|---:|---:|---|
| **VideoMamba Raw [O]** | RGB, offline; 32/8, 16 sampled frames | **83.9836** | **80.6135** | — | — | — | **B — khác causal protocol** |
| VideoMamba R3 [O, test-tuned] | RGB, offline + postprocess | 83.9940 | 80.6254 | — | — | — | B; test-tuned |
| RSC-GestureNet [P, Li et al. 2026] | Pose + RGB, causal | 93.33±0.24 | 91.71±0.27 | 91.69±0.29 | 98.80±0.07 | 0.153±0.013 | **B — không xếp hạng với Ours** |
| RSC-Gesture-Pose+VC [P] | Pose, causal, calibrated | 93.09±0.26 | 91.35±0.43 | 91.33±0.39 | 98.41±0.34 | 0.220±0.060 | B |
| TCN+VC [P] | Dữ liệu đặc trưng từ pose/RGB; causal | 92.97±0.18 | 91.16±0.21 | 91.08±0.23 | 98.41±0.07 | 0.214±0.007 | B |
| MD-GCN† tái triển khai trong RSC [P] | Pose, causal | 90.08±1.38 | 88.48±1.43 | 89.54±0.76 | 98.14±0.42 | 0.257±0.035 | B; **không phải** 98.95% từ Xiong 2021 |
| ST-GCN tái triển khai trong RSC [P] | Pose, causal | 89.45±0.65 | 87.94±0.72 | 82.64±0.43 | 98.06±0.41 | 0.328±0.060 | B |
| GRU baseline [P] | Temporal features, causal | 88.82±0.43 | 87.10±0.45 | 81.99±0.82 | 98.14±0.31 | 0.345±0.058 | B |

Nguồn: [Li, Gao & Fu, *RSC-GestureNet*, arXiv:2608.02200 (2026), Table 1](https://arxiv.org/abs/2608.02200), [bản HTML/toàn văn trên ResearchGate](https://www.researchgate.net/publication/411183058_RSC-GestureNet_Reliability-Aware_Selective_Causal_Recognition_of_Chinese_Traffic_Police_Gestures), [repository chính thức](https://github.com/chengli24/rsc-gesturenet-prcv2026), [CTP notebook nhóm](../261015/VideoMambaTinyCTPGestureFinal.ipynb).

Đối với VideoMamba, các metric **không có ở RSC** nhưng đã đo bằng notebook: `Mean Temporal Overlap Raw = 0.653223`, `R3 = 0.658979`; `Mean Gesture Recall Raw = 94.8035%`, `R3 = 94.4513%`; `Mean per-video FrameAcc Raw = 83.4973%`. **Không** đặt Temporal Overlap/Recall này cạnh Online-F1 hoặc Early@10 vì khác định nghĩa. Early@10 của RSC là đo nhận đúng sau **10 bước đầu ra kể từ chuyển trạng thái lệnh**, **không phải** Accuracy khi mới xem 10% thời lượng gesture.

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
- [ ] Thu thập exact test protocol và số liệu Param/FLOPs nguồn gốc của các công trình đã chọn; điền `—` khi không xác thực.
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
