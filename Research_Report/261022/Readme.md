# Báo cáo nghiên cứu ngày 22/10/2026

## Cập nhật 28/10/2026 — kiểm toán 7 bài báo và chuẩn hóa bảng SOTA

- **Mục tiêu:** cập nhật bốn nhóm bảng benchmark IPN-Hand/CTPGesture v1 từ bài báo gốc; phân biệt rõ *published* và số tự đo; ưu tiên nghiên cứu 2023–2026 và giữ benchmark nền tảng.
- **Bỏ hoàn toàn cột latency/inference runtime** khỏi bảng đối chiếu vì không dùng cùng phần cứng, độ chính xác tính toán, input và phạm vi đo (model-only so với full pipeline). Không dùng runtime khác paper để kết luận phương pháp nào nhanh hơn.
- **Giữ TTC** trong bảng RSC *causal* vì TTC là **Time-to-Correct-Command**, độ trễ phát đúng hiệu lệnh theo định nghĩa đánh giá trực tuyến, **không phải GPU inference runtime**; chưa có TTC tương đương cho VideoMamba.
- **[P0 — SOTA audit và protocol, 27/10](P0_SOTA_AUDIT_20261027.md):** bản lịch sử kiểm toán ban đầu. Những chỗ còn nhắc runtime trong tài liệu P0 phục vụ truy nguyên kết quả cũ, không phải cột so sánh của báo cáo cập nhật này.
- **[P1 — Runbook Kaggle/notebook](P1_NOTEBOOK_RUNBOOK_20261027.md)** và **[tool P1 offline metrics/keyframes](p1_offline_metrics_keyframes.py)** được giữ nguyên. **Chưa chạy lại notebook** hoặc tái huấn luyện trong lần cập nhật bảng này.

## A. Tổng quan

- **Proposed baseline:** VideoMamba-Tiny nhận 16 RGB frames (224×224); tách riêng **isolated** (đã biết GT gesture boundaries) và **continuous** (sliding windows trên full video).
- **IPN-Hand:** isolated 1.101 GT gesture clips; continuous 52 test videos; window 64, stride 16; các kết quả từ [IPN notebook](../261007/videomamba_man_configsBenchmark.ipynb).
- **CTPGesture v1:** isolated 1.728 GT segments (gồm 869 no-gesture); continuous 11 test videos; window 32, stride 8; các kết quả từ [CTP notebook](../261015/VideoMambaTinyCTPGestureFinal.ipynb).
- **Đã khảo sát chi tiết 7 công trình:** CoboGesture, Kausar, Gil-Martín, IPN-Hand original, RSC-GestureNet, Zhang, Xiong. Chưa có căn cứ cho tuyên bố *fair SOTA comparison* giữa tất cả công trình.

### Quy ước bằng chứng và mức so sánh

- **[O]** số đo/output trong notebook nhóm; **[P]** số công bố trong paper/benchmark gốc; **[E]** đếm/benchmark kiến trúc do nhóm thực hiện, **không phải số tác giả đối chứng công bố**; **[R]** thông số tham khảo của một *model family* khác, chưa khẳng định đúng checkpoint/bộ xử lý trong paper; **NR / —** không có số đúng cấu hình được xác minh.
- **A có điều kiện:** cùng dataset và metric gần tương đương, vẫn phải khóa test IDs, annotation version, preprocessing, aggregation và sampling; **B:** cùng nguồn dữ liệu nhưng khác split, modality hoặc giao thức; **C:** khác dataset version/tác vụ/metric, chỉ phục vụ Related Work. **Không dùng A có điều kiện hoặc B như một bảng thứ hạng SOTA đã được chứng minh.**
- **Parameters recognizer** là phần mô hình nhận dạng; **Parameters full pipeline** phải tính thêm detector/pose/flow/segmenter nếu dùng. **GFLOPs** chỉ có nghĩa khi cùng định nghĩa phép tính, độ phân giải, số frame và phạm vi model/pipeline; không tự chuyển MFLOPS, checkpoint size hoặc model size thành Params.
- Nơi có dấu **±**, giữ nguyên định nghĩa thống kê của tác giả (seeds, folds hoặc confidence intervals) và không mặc định cùng điều kiện với kết quả một checkpoint của nhóm.

### Danh mục bài báo được kiểm toán

| ID | Bài báo và nguồn gốc | Dataset / đánh giá | Trạng thái với VideoMamba |
|---|---|---|---|
| IPN-1 | [CoboGesture, IEEE Access 2026](https://doi.org/10.1109/ACCESS.2026.3664027), Tables 4, 7 | IPN-Hand isolated + continuous, RGB VideoMAE/v2 | **A có điều kiện — đối chứng gần nhất** |
| IPN-2 | [Kausar et al., Computers, Materials & Continua 2026](https://doi.org/10.32604/cmc.2026.084514), Tables 2, 3, 6–7 | IPN-Hand, multi-stage RGB/optical-flow/geometry + Transformer; **92.34%** | **B — chưa khóa test split** |
| IPN-3 | [Gil-Martín et al., ICAART 2025](https://doi.org/10.5220/0013053500003890), Table 1 p.28 | IPN-Hand, MediaPipe landmarks + LSTM; subject-wise CV | **B — khác giao thức đánh giá** |
| IPN-4 | [IPN-Hand original, ICPR 2020/21](https://doi.org/10.1109/ICPR48806.2021.9412317), Tables IV, V, IX | IPN-Hand original baseline; RGB/Flow/Seg | **A có điều kiện (isolated), khác metric ở continuous** |
| CTP-1 | [RSC-GestureNet, 2026](https://arxiv.org/abs/2608.02200), Section 4.1/Table 1 p.8–9 | Official CTPGesture v1; pose+RGB, full-frame causal | **B với VideoMamba offline — đối chứng continuous ưu tiên** |
| CTP-2 | [Zhang et al., Electronics 2026](https://doi.org/10.3390/electronics15122578), Tables 1–4 | 20 traffic-police videos; 10-video test, sequence recognition | **B — khác số video test và task** |
| CTP-3 | [Xiong et al., Electronics 2021](https://doi.org/10.3390/electronics10050551), Tables 1–3 | 9-video isolated test / 1.565 clips; skeleton + baselines RGB | **B — benchmark lịch sử, khác split** |

---

## Bảng 1 — IPN-Hand: Isolated Gesture Recognition

**Giao thức nhóm:** 1.101 GT gesture clips, loại class no-gesture; 16 RGB frames/clip, 224×224. VideoMamba [O] và kết quả [P] từ bài báo khác chưa xác nhận hoàn toàn cùng test IDs/preprocessing. **Không có cột GPU inference time.**

| Method | Năm | Input / đơn vị đánh giá | Params recognizer (M) | Params full pipeline (M) | GFLOPs/clip | Accuracy (%) | Mức | Nguồn |
|---|---:|---|---:|---:|---:|---:|---|---|
| **VideoMamba-Tiny (OURS) [O]** | 2026 | RGB, 16×224²; 1.101 GT clips | **6.9608** | **6.9608** | **29.0261 [E]** | **84.65** | Reference | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + **VideoMAEv2 [P]** | 2026 | RGB, isolated IPN | **303.8** | 303.8 recognizer only | **596.8327 [E]** | **84.59** | A có điều kiện | [Cobo Table 4](https://doi.org/10.1109/ACCESS.2026.3664027) · [GFLOPs benchmark nhóm](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAE [P] | 2026 | RGB, isolated IPN | **303.8** | 303.8 recognizer only | **596.8327 [E]** | **78.81** | A có điều kiện | [Cobo Table 4](https://doi.org/10.1109/ACCESS.2026.3664027) |
| **Kausar et al. [P]** | 2026 | RGB + segmentation/flow/geometry + Transformer | NR | NR | NR — Table 2 báo chi phí các module, chưa xác minh tổng | **92.34** | B: chưa khóa split | [Kausar Tables 2–3, 7](https://doi.org/10.32604/cmc.2026.084514) |
| **Gil-Martín et al. [P]** | 2025 | MediaPipe 21 landmarks + LSTM; 10-fold subject-wise CV | **0.058614** | NR (thiếu MediaPipe) | NR | **84.66 ± 1.09** | B: khác split | [Gil-Martín Table 1, p.28](https://www.scitepress.org/Papers/2025/130535/130535.pdf) |
| Gil-Martín, 37/13 subjects [P] | 2025 | MediaPipe + LSTM; original-style subject split | **0.058614** | NR (thiếu MediaPipe) | NR | **86.38 ± 2.03** | B: IDs chưa khóa | [Gil-Martín Sec. 3, p.28](https://www.scitepress.org/Papers/2025/130535/130535.pdf) |
| ResNeXt-101 RGB+Flow [P] | 2020/21 | RGB + SPyNet, 32f; official IPN | 47.56 | **49.000 [E]** | NR | **86.32** | B: khác modality/input | [IPN Tables IV–V](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 RGB+Seg [P] | 2020/21 | RGB + HarDNet, 32f; official IPN | 47.56 | **51.674 [E]** | NR | **84.77** | B: khác modality/input | [IPN Tables IV–V](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 RGB [P] | 2020/21 | RGB, 32f; official IPN | 47.51 | 47.51 | NR | **83.59** | A có điều kiện | [IPN Table IV](https://gibranbenitez.github.io/IPN_Hand/Results.html) |

**Chú thích Bảng 1:** [E] 596.8327 và 29.0261 là **ước lượng từ phép đo kiến trúc của nhóm ở FP32/T4**, input 16×224², quy ước 1 MAC = 1 FLOP-equivalent, **không phải số published GFLOPs của Cobo**. 49.000M/51.674M là tổng recognizer + module SPyNet 1.440M / HarDNet 4.114M từ IPN Table V, không phải tổng tác giả báo nguyên dạng. Giá trị 0.058614M của Gil-Martín chỉ là LSTM recognizer. Kausar có số theo **module**, chưa xác nhận Params/GFLOPs toàn pipeline nên để NR. Các hàng ResNet-50 và C3D bổ sung xem [IPN original leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) thay vì lặp quá nhiều kiến trúc 2020/21 trong bảng SOTA gần đây.

## Bảng 2 — IPN-Hand: Continuous Gesture Recognition

### Bảng 2A — Mean frame accuracy và temporal overlap (CoboGesture)

| Method | Input / protocol | Params recognizer (M) | Params full pipeline (M) | Mean per-video FrameAcc (%) | Temporal overlap | Gesture Recall (%) | Mức | Nguồn |
|---|---|---:|---:|---:|---:|---:|---|---|
| **VideoMamba Raw (OURS) [O]** | RGB, offline, window 64/step 16 | **6.9608** | **6.9608** | **83.4889** | **0.491110** | **78.8995** | Reference | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| VideoMamba R3 (test-tuned) [O] | RGB, offline; ngưỡng được thử trên test | 6.9608 | 6.9608 | 83.5249 | 0.507705 | 76.2844 | **Không phải independent test** | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAEv2 [P] | RGB, offline; window 64/step 30 | 303.8 | 303.8 recognizer only | **83.00** | **0.510** | NR | A có điều kiện | [Cobo Table 7](https://doi.org/10.1109/ACCESS.2026.3664027) |
| RGB-CoGes + VideoMAE [P] | RGB, offline; window 64/step 30 | 303.8 | 303.8 recognizer only | **80.00** | **0.370** | NR | A có điều kiện | [Cobo Table 7](https://doi.org/10.1109/ACCESS.2026.3664027) |

**Lưu ý:** 83.4889 là **mean per-video FrameAcc**, không thay bằng global FrameAcc **83.7308** khi đối chiếu Cobo 83.00. Temporal Overlap và Gesture Recall của nhóm dùng cách matching theo tolerance 2 giây; chỉ xếp hạng overlap với Cobo sau khi xác minh cùng công thức. R3 làm recall giảm so với Raw và được chọn sau khảo sát trên test.

### Bảng 2B — IPN-Hand original continuous Levenshtein benchmark

**Bảng lịch sử riêng:** Levenshtein Accuracy không phải FrameAcc; VideoMamba hiện chưa có Levenshtein Accuracy theo đúng official IPN protocol. Không so hai đại lượng với nhau.

| Method | Input | Params classifier (M) | Params pipeline (M) | Levenshtein Accuracy (%) | Nguồn |
|---|---|---:|---:|---:|---|
| ResNeXt-101 + ResLight + SPyNet | RGB+Flow | 47.56 | **49.908 [E]** | **42.47** | [IPN Table IX, leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight + HarDNet | RGB+Seg | 47.56 | **52.582 [E]** | **39.01** | [IPN Table IX, leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight | RGB | 47.51 | **48.405 [E]** | **25.34** | [IPN Table IX, leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |

**Chú thích:** Params pipeline [E] cộng recognizer, ResLight detector (~0.895M) và SPyNet/HarDNet khi sử dụng; đây là phép cộng từ module, không phải Params do Table IX công bố nguyên dạng.

---

## Bảng 3 — CTPGesture v1: Isolated / clip classification

**Giao thức nhóm:** official 11 test videos, 1.728 GT segments (869 no-gesture), chín nhãn 0–8, uniform sample 16 RGB frames/clip; VideoMamba isolated Accuracy **98.5532%**, Macro-F1 **97.9590%** trên cả chín lớp. Nếu chỉ xét 859 GT gesture segments (nhãn 1–8), Accuracy **98.02%**. Xiong dùng 9 test videos/1.565 clips, không trực tiếp xếp hạng với nhóm.

| Method | Năm | Input / đơn vị đánh giá | Params recognizer (M) | Params full pipeline (M) | GFLOPs/clip | Accuracy (%) | Mức | Nguồn |
|---|---:|---|---:|---:|---|---:|---|---|
| **VideoMamba-Tiny (OURS) [O]** | 2026 | RGB 16×224²; 1.728 GT segments; 9 lớp | **6.959817** | **6.959817** | **29.0261 [E]** | **98.5532** | Reference | [CTP notebook cell 16/26](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| **GSE + MD-GCN (Xiong) [P]** | 2021 | RGB→OpenPose→skeleton, 80f; 1.565 clips | NR | NR | NR | **98.95** | B: khác split/modality | [Xiong Tables 1, 3](https://doi.org/10.3390/electronics10050551) |
| P3D (Qiu; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | NR (model-family ≈65.3 [R]) | NR | NR | **92.21** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) |
| C3D (Tran; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | NR (model-family 78.4 [R]) | NR | NR (model-family 38.5 [R] ở 16×112²) | **87.12** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) |
| 3D ResNet (Hara; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | NR (R3D-18 ≈33.37 [R]) | NR | NR | **81.16** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) |

**Quan trọng:** Xiong Table 1 ghi **Model Size = 11.7 M** và một giá trị *Inference Time = 3.0 s* chưa rõ đơn vị thao tác. **11.7 M không được mặc nhiên coi là 11.7 triệu Params**, và **không chép thời gian đó vào bảng**. Thông số [R] cho P3D/C3D/R3D tham khảo từ model family khác, **không phải đúng variant/checkpoint của Xiong**; không sử dụng để khẳng định ưu thế tính toán.

### Bảng 3B — Cùng lĩnh vực CTPGesture nhưng không cùng isolated protocol

| Method | Năm | Input / protocol | Params recognizer (M) | Params full pipeline (M) | GFLOPs/sequence | Accuracy published (%) | Nguyên nhân không so thứ hạng | Nguồn |
|---|---:|---|---:|---|---|---:|---|---|
| **YOLO11m-Pose → BiLSTM–Transformer (Zhang) [P]** | 2026 | RGB→17 pose joints→40-frame sequence; 10 test videos | **0.08641** | NR (cần xác minh detector đúng variant) | NR | **98.91** | 10 video test, sequence-to-one; thiếu video 101 so với nhóm | [Zhang Tables 1–4](https://doi.org/10.3390/electronics15122578) |

**Chú thích:** Zhang Table 3 báo Params **86.41K** chỉ cho recognition head; không gộp thành tổng Params hệ thống. Có sự không nhất quán giữa Table 3 và phần diễn giải về BiLSTM standalone (96.48% so với 98.91%); ưu tiên số trong bảng gốc và ghi chú khi trích dẫn. Một số nghiên cứu traffic-police khác có cùng tên lớp nhưng chưa khóa được official CTPGesture v1 split; không điền thêm bằng Accuracy chưa kiểm chứng.

---

## Bảng 4 — CTPGesture v1: Continuous Gesture Recognition

**Tách bắt buộc:** VideoMamba hiện dùng **offline overlapping-window frame aggregation**; RSC dùng **causal end-of-window** và stabilization. Accuracy/Macro-F1 của hai bảng dưới đây khác evaluation protocol, nên **không xếp hạng trực tiếp**.

### Bảng 4A — VideoMamba offline trên full video (nhóm tự đo)

| Method | Window / Input | Params recognizer (M) | Global FrameAcc (%) | Global Macro-F1 (%) | Mean per-video FrameAcc (%) | Mean temporal overlap | Mean gesture recall (%) | Nguồn |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **VideoMamba Raw [O]** | RGB 32/8, 16×224² | **6.959817** | **83.9836** | **80.6135** | **83.4973** | **0.653223** | **94.8035** | [CTP notebook cell 22](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| VideoMamba R3 (test-tuned) [O] | RGB 32/8, 16×224² | 6.959817 | **83.9940** | **80.6254** | NR | **0.6590** | **94.45** | [CTP notebook cell 24](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |

R3 dùng post-processing được lựa chọn sau khảo sát test; không trình bày như *unseen test*. Bảng này không có Online-F1/TTC hợp lệ; không suy chúng từ offline cache.

### Bảng 4B — RSC-GestureNet và các baseline causal (do tác giả RSC công bố)

| Method | Input / protocol | Params recognizer (M) | Accuracy (%) | Macro-F1 (%) | Online-F1 (%) | Early@10 (%) | **TTC (s)** | Robust-F1 (%) | Nguồn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **RSC-GestureNet (Ours) [P]** | Pose+RGB, causal, calibrated | **≥0.815270 [E]*** | **93.33±0.24** | **91.71±0.27** | **91.69±0.29** | **98.80±0.07** | **0.153±0.013** | **90.97±0.37** | [RSC Table 1, p.9](https://arxiv.org/pdf/2608.02200) |
| RSC-GestureNet-Fusion [P] | Pose+RGB, causal | **0.815270 [E]** | 91.38±0.38 | 89.95±0.42 | 89.28±0.53 | 98.72±0.20 | 0.191±0.040 | 88.93±0.77 | [RSC Table 1](https://arxiv.org/pdf/2608.02200) |
| RSC-Gesture-Pose+VC [P] | Pose, causal, calibrated | ≈0.335531 [E] | 93.09±0.26 | 91.35±0.43 | 91.33±0.39 | 98.41±0.34 | 0.220±0.060 | 89.94±0.26 | [RSC Table 1](https://arxiv.org/pdf/2608.02200) |
| TCN+VC [P] | Pose, causal, calibrated | NR | 92.97±0.18 | 91.16±0.21 | 91.08±0.23 | 98.41±0.07 | 0.214±0.007 | 90.29±0.34 | [RSC Table 1](https://arxiv.org/pdf/2608.02200) |
| ST-GCN+VC [P] | Pose, causal, calibrated | NR | 92.47±0.39 | 90.73±0.50 | 90.63±0.57 | 98.21±0.18 | 0.229±0.034 | 87.58±0.79 | [RSC Table 1](https://arxiv.org/pdf/2608.02200) |
| MD-GCN-style reimplemented [P] | Pose, causal | 0.399993 [E] | 90.08±1.38 | 88.48±1.43 | 89.54±0.76 | 98.14±0.42 | 0.257±0.035 | NR | [RSC Table 1 / baseline code](https://github.com/chengli24/rsc-gesturenet-prcv2026) |
| HLP-GCN-style reimplemented [P] | Pose, causal | 0.335817 [E] | 89.08±0.22 | 87.60±0.33 | 88.62±1.16 | 98.10±0.29 | 0.313±0.149 | NR | [RSC Table 1 / baseline code](https://github.com/chengli24/rsc-gesturenet-prcv2026) |

**Ghi chú bắt buộc cho Bảng 4B:**
- **TTC (Time-to-Correct-Command)** đo khoảng từ chuyển trạng thái sang lệnh đúng đầu tiên ở các bước dự đoán nhân quả, tính theo 15 FPS, **không phải runtime phần cứng**. Early@10 của RSC là 10 *output steps* sau transition, **không phải prefix Accuracy@10%** của P1.
- RSC Table 1 công bố **mean ± standard deviation của ba random seeds** trên official CTPGesture v1, window 32/stride 4, 17 joints; GPU nghiên cứu là RTX A4000 nhưng không dùng GPU này để so runtime với T4.
- [E] 0.815270M chỉ là recognizer *fusion* nhóm đếm dựa trên [source RSC](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py). Hàng RSC-GestureNet **(Ours)** có thêm calibration/branch-selection nên **không được gọi 0.815270M là exact full pipeline**. Tất cả các hàng dựa trên pose chưa bao gồm MediaPipe pose extractor trong số Params.
- `MD-GCN-style reimplemented` là mã baseline **do nhóm tác giả RSC tái triển khai**, không phải đúng original MD-GCN Xiong 2021; không đánh đồng Accuracy 90.08% causal với 98.95% isolated.

---

## B. Khó khăn còn tồn tại

1. Chưa xác minh exact test manifest/preprocessing của CoboGesture, Kausar và phép 37/13 subjects ở Gil-Martín; Zhang chỉ dùng 10 test videos so với 11 của nhóm.
2. Nhiều bài không công bố Params/GFLOPs **đúng toàn pipeline**. Không điền ước lượng model family như số published chính xác.
3. VideoMamba mới có **một checkpoint**, continuous hiện là **offline**, và cấu hình R3 được thử chọn trên test; chưa có evaluation theo cùng causal protocol với RSC.

## C. Việc tiếp theo phục vụ publication

1. **Khóa provenance:** DOI, bảng/trang gốc, test IDs, class mapping, split, annotation, preprocessing và định nghĩa metric cho từng hàng; những ô NR vẫn để nguyên nếu paper không công bố.
2. **Chuẩn hóa complexity nội bộ:** rà soát việc báo 17–29 GFLOPs ở các bản báo cáo trước; chốt một quy ước FLOPs/MAC, ghi tool/precision/input.
3. **Chạy fair baseline:** dùng cùng split/evaluator để đánh giá một số kiến trúc RGB (VideoMamba, một 3D CNN hiệu quả và một Video Transformer). Muốn so trực tiếp với RSC phải xây evaluator causal.
4. **Không thêm lại latency vào bảng chỉ từ tài liệu công bố:** chỉ lập bảng thời gian hệ thống riêng khi tất cả model được benchmark lại trong điều kiện đồng nhất.
