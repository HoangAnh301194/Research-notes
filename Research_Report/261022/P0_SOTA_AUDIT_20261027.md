# P0 — Tổng hợp đối chứng SOTA và kiểm toán giao thức (27/10/2026)

**Mục tiêu:** Hoàn tất nhóm việc P0 đã giao ngày 08, 15, 17, 22/10/2026: (i) hoàn thiện 4 bảng isolated/continuous cho IPN-Hand và CTPGesture v1; (ii) kiểm tra khả năng đối chiếu, (iii) khảo sát đối chứng gần đây, (iv) làm bảng tổng quan có Parameters/GFLOPs/metrics và link cho thầy. **Không chạy lại model, không sửa checkpoint.**

**Bảng số liệu đang được duy trì:** [4 bảng benchmark trong báo cáo 22/10](Readme.md). Tài liệu này là **bảng kiểm toán về nguồn, tính công bằng và phạm vi**, không thay thế output notebook.

## 1. Kết luận quan trọng trước khi đọc các con số

- **IPN-Hand:** CoboGesture 2026 là bài đối chứng *RGB + isolated + continuous* gần nhất. Cobo báo isolated **84.59%** với RGB-CoGes + VideoMAEv2; continuous **83.00% mean per-video Frame-wise Accuracy**, **0.510 overlap**. VideoMamba IPN có **84.65% isolated**, **83.4889% raw mean per-video FrameAcc**, **0.491110 overlap**; R3 tăng FrameAcc **83.5249%** nhưng **được dò ngưỡng trên test**. Chưa có bằng chứng thống kê hay giao thức hoàn toàn đồng nhất để kết luận vượt SOTA. [CoboGesture 2026](https://doi.org/10.1109/ACCESS.2026.3664027) · [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb).
- **CTPGesture isolated:** VideoMamba **98.5532%**, **97.9590% Macro-F1**, từ **1.728 segments/11 test videos**, bao gồm class 0. MD-GCN Xiong 2021 báo **98.95%** ở **subset test khác (9 videos/1.565 samples)** và dùng skeleton; **không so trực tiếp thứ hạng**. [Xiong 2021](https://www.mdpi.com/2079-9292/10/5/551) · [Notebook CTP](../261015/VideoMambaTinyCTPGestureFinal.ipynb).
- **CTPGesture continuous:** VideoMamba Raw offline gán nhãn frame theo *all overlapping windows*, **83.9836% global FrameAcc**, **80.6135% Macro-F1**. RSC-GestureNet 2026 dùng **causal/end-of-window**, pose + RGB descriptors, báo **93.33±0.24% accuracy, 91.71±0.27% Macro-F1, 91.69±0.29% Online-F1**. **Không so 83.98 với 93.33 như cùng metric/protocol.** [RSC 2026](https://arxiv.org/abs/2608.02200) · [RSC source](https://github.com/chengli24/rsc-gesturenet-prcv2026) · [Notebook CTP](../261015/VideoMambaTinyCTPGestureFinal.ipynb).
- **Khối lượng tham số:** IPN VideoMamba 6.9608M; CTP VideoMamba 6.959817M; VideoMAE/VideoMAEv2 ViT-L ~303.8M. **29.0261 GFLOPs VideoMamba và 596.8327 GFLOPs VideoMAE là phép đếm kiến trúc của nhóm**, không phải FLOPs đồng nhất từ hai bài báo; những pipeline dựa trên pose phải tính cả bộ trích keypoints nếu so RGB-to-command. [Notebook complexity IPN](../261007/videomamba_man_configsBenchmark.ipynb) · [Notebook complexity CTP](../261015/VideoMambaTinyCTPGestureFinal.ipynb).

## 2. Ma trận đối chứng theo paper, input và mức công bằng

**Ký hiệu:** A = cùng nhiệm vụ, metric gần tương thích, còn phải kiểm tra chi tiết split; B = cùng hoặc gần dataset nhưng khác split/modality/task/aggregation; C = related work, không so điểm số trực tiếp; N/A = không dùng dataset đang xét.

| Công trình | Năm; nơi công bố | Dataset/đơn vị | Input | Đối chiếu được ở bảng | Mức | Kết luận và nguồn |
|---|---|---|---|---|---|---|
| **CoboGesture — RGB-CoGes + VideoMAEv2** | 2026; IEEE Access | IPN-Hand isolated + continuous | RGB | IPN Bảng 1, Bảng 2A | **A** | 84.59% isolated; mean FrameAcc 83.00%, overlap 0.510. [Paper, Tables 4/7](https://doi.org/10.1109/ACCESS.2026.3664027) |
| **IPN-Hand original benchmark** | 2020/21; ICPR | IPN-Hand original | RGB, RGB+Flow, RGB+Seg | IPN Bảng 1, Bảng 2B | **B** | RGB ResNeXt-101 isolated 83.59%; continuous báo **Levenshtein Accuracy**, không phải FrameAcc. [Leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| **Multimodal HGR — Benitez-Garcia et al.** | 2024; SoMeT | IPN-Hand **đã chỉnh annotation / 14 classes** | RGB + **synthetic depth/flow** | Related IPN | **C** | Không so với IPN original chưa sửa nhãn. [DOI](https://doi.org/10.3233/FAIA240397) |
| **IPN HandS** | 2025; Applied Sciences | IPN HandS, **refined temporal annotations + skeleton** | Pose/skeleton | Related IPN | **C** | Annotation đổi; không tráo benchmark với RGB original. [Paper](https://www.mdpi.com/2076-3417/15/11/6321) |
| **Continuous HGR: Benchmarks and Methods** | 2025; CVIU | Tổng quan, có IPN-Hand | Nhiều | Related IPN | **C** | Có nghiên cứu dùng Levenshtein, tổng quan không tương đương RGB isolated Accuracy. [Bài khảo sát](https://doi.org/10.1016/j.cviu.2025.104435) |
| **Xiong et al., GSE+MD-GCN** | 2021; Electronics | CTPGesture v1, **9-video isolated subset** | Skeleton/GCN | CTP Bảng 3 | **B** | Accuracy 98.95% trên subset khác; model size 11.7M chưa rõ đơn vị Params. [Paper](https://www.mdpi.com/2079-9292/10/5/551) |
| **RSC-GestureNet — Li et al.** | 2026; PRCV Oral/preprint | CTPGesture v1 **causal window** | Pose+RGB descriptors | CTP Bảng 4 | **B** | Accuracy 93.33±0.24, Macro-F1 91.71±0.27, Online-F1 91.69±0.29; **khác offline per-frame protocol**. [Paper](https://arxiv.org/abs/2608.02200) · [Code](https://github.com/chengli24/rsc-gesturenet-prcv2026) |
| **Zhang et al., BiLSTM–Transformer** | 2026; Electronics | Police gesture **20 videos, 10 video test**, sequence-to-one | RGB→YOLO11m-Pose | Related CTP isolated | **B** | Recognition head 86.41K, detector chưa gồm; 98.91% **không phải 1.728-segment test**. [Paper](https://www.mdpi.com/2079-9292/15/12/2578) |
| **Xiao et al., NTPGR** | 2025; Scientific Reports | Traffic-police gestures **448 video sequences** | RGB→keyframe/pose | Related CTP | **C** | Accuracy 97.56% gesture-instance, **không khẳng định cùng official CTPGesture v1 split**. [Paper](https://www.nature.com/articles/s41598-025-02833-y) |
| **FDMB-YOLOv11** | 2026; PLOS ONE | **3.728 ảnh** lấy từ video/augment ánh sáng | Single-frame RGB detection | Related CTP | **C** | 1.93M Params nhưng metric **mAP**, không phải video Accuracy/Macro-F1. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13524292/) |
| **Chinese Traffic Police Gesture Recognition Based on GCN in Natural Scene** | 2021; Applied Sciences | **20.480 video tự thu thập**, 5 camera | Skeleton | Related CTP | **C** | **Dataset khác** dù 8 tên cử chỉ tương tự. [Paper](https://www.mdpi.com/2076-3417/11/24/11951) |

**Kết luận phần khảo sát:** Đã tìm được trên 5 công trình có chủ đề traffic-police gesture để thầy nhìn tổng quan, nhưng **chưa có 5 phương pháp RGB-only 2023–2026 được xác minh cùng official CTPGesture v1 + split + isolated protocol**. Không được biến danh sách related work thành 5 hàng so sánh SOTA hợp lệ. **Hai ứng viên benchmark chính:** CoboGesture 2026 và RSC-GestureNet 2026; các phương pháp còn lại hỗ trợ contextual literature, lịch sử và thảo luận.

## 3. Ma trận protocol cần khóa trước khi viết kết luận / chạy tiếp

| Trường | IPN ours | IPN Cobo | CTP ours | CTP RSC |
|---|---|---|---|---|
| Dataset | IPN original, test annotated | IPN-Hand, xác minh annotation/split | CTPGesture v1, 11 official test IDs | CTPGesture v1 official split |
| Đầu vào | 16 RGB/clip; continuous 64f→16f, stride16 | RGB-CoGes/VideoMAE | 16 RGB/clip; continuous 32f→16f, stride8 | 17-joint pose + RGB descriptors, causal 32f stride4 |
| Mục tiêu isolated | 1.101 GT gesture clips; bỏ no-gesture | Theo protocol Cobo Table 4 | 1.728 GT segments (869 no-gesture) | N/A (chủ yếu continuous window) |
| Continuous | frame-level, offline aggregation | mean per-video frame + overlap | frame-level, offline aggregation | causal window-level + Online-F1 / TTC |
| Chọn hậu xử lý | R3 đã dò trên test | Paper thresholds | R3 đã dò trên test | Calibration trên train/val, 3-seed result |
| Tham số | 6.9608M model RGB | ~303.8M recognizer ViT-L | 6.959817M model RGB | 0.815270M recognizer fusion **chưa kể RGB→pose** |
| Mức fair comparison | **A có điều kiện** | **A có điều kiện** | **B** với RSC | **B** với ours |

**Trước khi kết luận vượt trội**, cần chứng minh: annotation version, official video IDs và class mapping trùng, sampling/frame-rate đúng, định nghĩa metric giống nhau, kết quả raw held-out không bị selection on test, và đủ kiểm định thống kê nếu độ chênh nhỏ. **Hạn chế quan trọng:** RSC báo kết quả `mean±std` qua 3 seeds; VideoMamba hiện đánh giá một checkpoint.

## 4. Trạng thái checklist P0 (kể cả công việc chưa thể xác nhận)

- [x] **4 bảng đối chiếu** đã có (IPN isolated, IPN continuous, CTP isolated, CTP continuous) tại [Readme.md](Readme.md), mỗi hàng có nguồn.
- [x] **Ma trận đối chứng A/B/C** và protocol đã thiết lập trong báo cáo này.
- [x] **Bảng tổng quan paper để gửi thầy** đã soạn, có nguồn và cảnh báo khác benchmark.
- [x] **Truy nguồn Parameters/GFLOPs** đã bổ sung vào báo cáo chính (các Params family-reference phải giữ ký hiệu ước lượng, không gán exact).
- [x] **Khảo sát IPN RGB/non-skeleton**: ưu tiên Cobo 2026, original RGB; ghi rõ các bài 2024/2025 dùng refined annotations/skeleton/derived modalities.
- [ ] **Chưa đạt mục tiêu 5 bài RGB-only cùng CTP v1 official split, 2023–2026**: sau khảo sát không xác minh đủ, không được tự gắn nhãn đạt.
- [ ] **Chưa xác minh hành động gửi cho thầy**: đã chuẩn bị tài liệu, chưa có bằng chứng đã gửi.
- [ ] **Chưa khóa manifest/checkpoint SHA và split exact từng paper**; phải hoàn thành trước tuyên bố fair SOTA.

## 5. Quyết định cho bước P1

**Không cần chạy lại training.** Hai notebook hiện đều có output CSV isolated và raw probability cache continuous: IPN `videomamba_ipn_isolated_test.csv`, `videomamba_postprocess_raw_cache.pkl` (cell 12,18); CTP `isolated_predictions.csv`, `continuous_raw_cache.pkl` (cell 16,22). Nếu còn các file này, tính lại **classification metrics, Levenshtein edit-score, segment confusion, lỗi keyframe** bằng CPU. Chỉ cần inference bổ sung cho **Accuracy@10/20/50%**, vì cần cấp model đoạn video **chỉ đến mốc 10/20/50%**, không thể suy ra bằng phép cắt cửa sổ full-context; **Early@10% là metric khác Early@10 output steps của RSC**.

Xem hướng dẫn P1 và thứ tự chạy notebook tại [P1_NOTEBOOK_RUNBOOK_20261027.md](P1_NOTEBOOK_RUNBOOK_20261027.md).
