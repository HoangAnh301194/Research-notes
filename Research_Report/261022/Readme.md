# Báo cáo nghiên cứu ngày 22/10/2026

## A. Công việc đã làm
- Khảo sát các bài báo nghiên cứu gần đây có sử dụng IPN Hand và CTPGesture 

### Quy ước kí hiệu 

- **[O]** số đo/output trong notebook nhóm; **[P]** số công bố trong paper/benchmark gốc; **[E]** đếm/benchmark kiến trúc do nhóm thực hiện, **không phải số tác giả đối chứng công bố**; **[R]** Params/GFLOPs **tham chiếu của model family/variant** từ paper gốc hoặc model zoo (chưa xác nhận trùng checkpoint tác giả); **[D]** phép cộng các module đã biết (không phải số tổng paper công bố); **NR / —** chưa xác minh đúng cấu hình.

- **A có điều kiện:** cùng dataset và metric gần tương đương, vẫn phải khóa test IDs, annotation version, preprocessing, aggregation và sampling; **B:** cùng nguồn dữ liệu nhưng khác split, modality hoặc giao thức; **C:** khác dataset version/tác vụ/metric, chỉ phục vụ Related Work.

- **Parameters recognizer** là phần mô hình nhận dạng; **Parameters full pipeline** phải tính thêm detector/pose/flow/segmenter nếu dùng.

### Danh sách bài báo được khảo sát

| ID | Bài báo và nguồn gốc | Dataset / đánh giá | Trạng thái với VideoMamba |
|---|---|---|---|
| IPN-1 | [CoboGesture, IEEE Access 2026](https://doi.org/10.1109/ACCESS.2026.3664027), Tables 4, 7 | IPN-Hand isolated + continuous, RGB VideoMAE/v2 | **A có điều kiện — đối chứng gần nhất** |
| IPN-2 | [Kausar et al., Computers, Materials & Continua 2026](https://doi.org/10.32604/cmc.2026.084514), Tables 2, 3, 6–7 | IPN-Hand, multi-stage RGB/optical-flow/geometry + Transformer; **92.34%** | **B — chưa khóa test split** |
| IPN-3 | [Gil-Martín et al., ICAART 2025](https://doi.org/10.5220/0013053500003890), Table 1 p.28 | IPN-Hand, MediaPipe landmarks + LSTM; subject-wise CV | **B — khác giao thức đánh giá** |
| IPN-4 | [IPN-Hand original, ICPR 2020/21](https://doi.org/10.1109/ICPR48806.2021.9412317), Tables IV, V, IX | IPN-Hand original baseline; RGB/Flow/Seg | **A có điều kiện (isolated), khác metric ở continuous** |
| IPN-5 | [TY-Net, IEEE Access 2023](https://doi.org/10.1109/ACCESS.2023.3341702), Sec. V-A/Table 2 pp. 140389–140391 | IPN-Hand continuous, MediaPipe skeleton + Transformer, Levenshtein | **B với VideoMamba — khác metric; đối chứng sequence hiện đại** |
| IPN-6 | [Nayan et al., Machine Vision and Applications 2024](https://doi.org/10.1007/s00138-024-01565-9), Table 12, p. 12/15 | IPN-Hand continuous, RGB + Flow + EHTD + Seg, Levenshtein | **B — khác modality và metric** |
| IPN-7 | [TMMF, IEEE Transactions on Image Processing 2021](https://doi.org/10.1109/TIP.2021.3108349); [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9) | IPN-Hand continuous, RGB + Flow, Levenshtein | **B — benchmark lịch sử; khác metric** |
| IPN-8 | [TD-Net, ICCAIS 2022](https://doi.org/10.1109/ICCAIS56082.2022.9990122) | IPN-Hand continuous, skeleton detector + recognizers, Levenshtein | **B — benchmark lịch sử; khác metric** |
| CTP-1 | [RSC-GestureNet, 2026](https://arxiv.org/abs/2608.02200), Section 4.1/Table 1 p.8–9 | Official CTPGesture v1; pose+RGB, full-frame causal | **B với VideoMamba offline — đối chứng continuous ưu tiên** |
| CTP-2 | [Zhang et al., Electronics 2026](https://doi.org/10.3390/electronics15122578), Tables 1–4 | 20 traffic-police videos; 10-video test, sequence recognition | **B — khác số video test và task** |
| CTP-3 | [Xiong et al., Electronics 2021](https://doi.org/10.3390/electronics10050551), Tables 1–3 | 9-video isolated test / 1.565 clips; skeleton + baselines RGB | **B — benchmark lịch sử, khác split** |

---

## Bảng 1 — IPN-Hand: Isolated Gesture Recognition

**Protocol đánh giá proposed method:** 16 RGB frames/clip, 224×224. 

| Method | Năm | Input | Params recognizer (M) | Params full pipeline (M) | GFLOPs/clip | Accuracy (%) | Mức | Nguồn |
|---|---:|---|---:|---:|---:|---:|---|---|
| **VideoMamba-Tiny (OURS) [O]** | 2026 | RGB, 16×224²; 1.101 GT clips | **6.9608** | **6.9608** | **29.0261 [E]** | **84.65** | Reference | [Notebook IPN](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + **VideoMAEv2 [P]** | 2026 | RGB, isolated IPN | **303.8** | 303.8 recognizer only | **596.8327 [E]** | **84.59** | A có điều kiện | [Cobo Table 4](https://doi.org/10.1109/ACCESS.2026.3664027) · [GFLOPs benchmark nhóm](../261007/videomamba_man_configsBenchmark.ipynb) |
| RGB-CoGes + VideoMAE [P] | 2026 | RGB, isolated IPN | **303.8** | 303.8 recognizer only | **596.8327 [E]** | **78.81** | A có điều kiện | [Cobo Table 4](https://doi.org/10.1109/ACCESS.2026.3664027) |
| **Kausar et al. [P]** | 2026 | RGB + segmentation/flow/geometry + Transformer | NR (Transformer head) | **NR tổng**; ResNet-34 **21.797672 [R]** chỉ là backbone tham chiếu của segmentation U-Net | NR — Table 2 báo chi phí các module, chưa xác minh tổng | **92.34** | B: chưa khóa split | [Kausar Tables 2–3, 7](https://doi.org/10.32604/cmc.2026.084514) · [Torchvision ResNet-34](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34) |
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

### Bảng 2B — IPN-Hand: Continuous Levenshtein Accuracy (đối chứng 2023–2024 và baseline gốc)

**Chỉ so các phương pháp có kết quả Levenshtein trên IPN-Hand.** VideoMamba hiện mới có FrameAcc/Overlap từ offline full-video aggregation, **chưa có Levenshtein Accuracy**; không đưa 83.4889% FrameAcc vào cột Levenshtein. Mọi hàng khác modality, detector, post-processing hoặc cấu hình đánh giá đều chỉ so tham khảo cho đến khi khóa test manifest và evaluator.

| Method | Năm | Input / protocol | Params recognizer (M) | Params full pipeline (M) | GFLOPs recognizer | Levenshtein Accuracy (%) | Nguồn gốc kết quả |
|---|---:|---|---:|---:|---|---:|---|
| **Nayan et al. — ME + EHTD + BiLSTM [P]** | **2024** | RGB + Seg + Flow + EHTD; continuous | **NR full**; GoogLeNet **6.624904 [R]/backbone** | NR (multi-stream GoogLeNet + BiLSTM) | NR | **73.71** | [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9) · [GoogLeNet ref](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html) |
| Nayan et al. — ME + EHTD + BiLSTM [P] | 2024 | RGB + Flow + EHTD; continuous | **NR full**; GoogLeNet **6.624904 [R]/backbone** | NR (multi-stream GoogLeNet + BiLSTM) | NR | **69.40** | [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9) · [GoogLeNet ref](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html) |
| **TY-Net — Doždor et al. [P]** | **2023** | MediaPipe hand skeleton + Transformer; online boundary detection | **≈5 [P]** | NR (thiếu MediaPipe) | **0.18 [P]**, recognizer only | **60.46** | [TY-Net Sec. V-A, p.140390; Table 2 p.140391](https://doi.org/10.1109/ACCESS.2023.3341702) |
| **TMMF — Gammulle et al. [P]** | 2021 | RGB + Flow; continuous temporal multimodal fusion | NR | NR | NR | **68.12** | [TMMF paper](https://doi.org/10.1109/TIP.2021.3108349) · [Nayan Table 12](https://doi.org/10.1007/s00138-024-01565-9) |
| **TD-Net — Nguyen et al. [P]** | 2022 | RGB→MediaPipe skeleton; gesture detector + 2 classifiers | NR | NR | NR | **40.10** | [TD-Net, ICCAIS 2022, Results/Conclusion](https://doi.org/10.1109/ICCAIS56082.2022.9990122) |
| ResNeXt-101 + ResLight + SPyNet [P] | 2020/21 | RGB+Flow; continuous original benchmark | 47.56 | **49.908 [E]** | NR | **42.47** | [IPN Table IX, official leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight + HarDNet [P] | 2020/21 | RGB+Seg; continuous original benchmark | 47.56 | **52.582 [E]** | NR | **39.01** | [IPN Table IX, official leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |
| ResNeXt-101 + ResLight [P] | 2020/21 | RGB; continuous original benchmark | 47.51 | **48.405 [E]** | NR | **25.34** | [IPN Table IX, official leaderboard](https://gibranbenitez.github.io/IPN_Hand/Results.html) |

**Lưu ý đối chiếu:**
- **TY-Net:** 60.46% là kết quả test trong *Section V-A* cho cấu hình đề xuất; *Table 2* còn tổng hợp thí nghiệm nhiều random seeds/folds, **không được thay tùy tiện bằng số trung bình của một giao thức khác**. ≈5M Params và 0.18 GFLOPs là phát biểu ở *Section V-D*, không gồm MediaPipe, không cùng đơn vị input/tác vụ với GFLOPs/clip của VideoMamba.
- **Nayan:** 73.71% và 69.40% trích từ *Table 12* của bản toàn văn được truy cập, cần xác minh lần cuối trên PDF chính thức từ nhà xuất bản khi chuẩn bị submission. Không phải Accuracy isolated.
- **TMMF:** 68.12% được đối chiếu trong Nayan Table 12 và survey CVIU 2025, đồng thời liên kết đến [bài gốc IEEE TIP](https://doi.org/10.1109/TIP.2021.3108349); chưa kiểm chứng đầy đủ cấu hình Params/GFLOPs từ paper gốc.
- **TD-Net:** 40.10% là Levenshtein Accuracy trên IPN-Hand theo bài ICCAIS, không phải isolated Accuracy (84.98% trong một đánh giá khác).
- **IPN original:** Params pipeline [E] cộng recognizer với ResLight detector (~0.895M) và SPyNet/HarDNet nếu có; không phải tổng Params được Table IX công bố nguyên dạng.
- **Không có cột GPU latency/runtime:** phần cứng và phạm vi thời gian đo khác nhau. Nhóm cần chạy phép tính **Levenshtein trên predictions VideoMamba** trước khi tuyên bố xếp hạng continuous cùng các bài trong Bảng 2B.

---

## Bảng 3 — CTPGesture v1: Isolated / clip classification

**Giao thức nhóm:** official 11 test videos, 1.728 GT segments (869 no-gesture), chín nhãn 0–8, uniform sample 16 RGB frames/clip; VideoMamba isolated Accuracy **98.5532%**, Macro-F1 **97.9590%** trên cả chín lớp. Nếu chỉ xét 859 GT gesture segments (nhãn 1–8), Accuracy **98.02%**. Xiong dùng 9 test videos/1.565 clips, không trực tiếp xếp hạng với nhóm.

| Method | Năm | Input / đơn vị đánh giá | Params recognizer (M) | Params full pipeline (M) | GFLOPs/clip | Accuracy (%) | Mức | Nguồn |
|---|---:|---|---:|---:|---|---:|---|---|
| **VideoMamba-Tiny (OURS) [O]** | 2026 | RGB 16×224²; 1.728 GT segments; 9 lớp | **6.959817** | **6.959817** | **29.0261 [E]** | **98.5532** | Reference | [CTP notebook cell 16/26](../261015/VideoMambaTinyCTPGestureFinal.ipynb) |
| **GSE + MD-GCN (Xiong) [P]** | 2021 | RGB→OpenPose→skeleton, 80f; 1.565 clips | **NR**; paper báo Model Size = 11.7 M, **không phải #Params** | NR | NR | **98.95** | B: khác split/modality | [Xiong Tables 1, 3](https://doi.org/10.3390/electronics10050551) |
| P3D (Qiu; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | **≈65.3 [R]** cho P3D-199; chưa xác nhận variant Xiong | NR | NR | **92.21** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) · [P3D gốc Fig. 1](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf) |
| C3D (Tran; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | **78.4 [R]** (MMAction2 C3D, chưa xác nhận checkpoint Xiong) | NR | NR; **38.5 GFLOPs/16×112² [R]** model zoo | **87.12** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) · [MMAction2 C3D](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md) |
| 3D ResNet (Hara; Xiong baseline) [P] | 2021 | RGB, variant paper Xiong | **33.371472 [R]** (R3D-18, chưa xác nhận depth Xiong) | NR | NR; **40.70 GFLOPs/16×112² [R]** model zoo | **81.16** | B: khác split | [Xiong Table 3](https://doi.org/10.3390/electronics10050551) · [R3D-18 reference](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html) |

**Quan trọng:** Xiong Table 1 ghi **Model Size = 11.7 M** và một giá trị *Inference Time = 3.0 s* chưa rõ đơn vị thao tác. **11.7 M không được mặc nhiên coi là 11.7 triệu Params**, và **không chép thời gian đó vào bảng**. Thông số [R] cho P3D/C3D/R3D tham khảo từ model family khác, **không phải đúng variant/checkpoint của Xiong**; không sử dụng để khẳng định ưu thế tính toán.

### Bảng 3B — Cùng lĩnh vực CTPGesture nhưng không cùng isolated protocol

| Method | Năm | Input / protocol | Params recognizer (M) | Params full pipeline (M) | GFLOPs/sequence | Accuracy published (%) | Nguyên nhân không so thứ hạng | Nguồn |
|---|---:|---|---:|---|---|---:|---|---|
| **YOLO11m-Pose → BiLSTM–Transformer (Zhang) [P]** | 2026 | RGB→17 pose joints→40-frame sequence; 10 test videos | **0.08641 [P]** | **≈20.98641 [D/R]** = YOLO11m-Pose 20.9 [R] + head 0.08641 [P]; chưa tính phụ trợ | NR; **71.5 GFLOPs/frame @640 [R]** cho pose detector, không phải GFLOPs/sequence | **98.91** | 10 video test, sequence-to-one; thiếu video 101 so với nhóm | [Zhang Table 3](https://doi.org/10.3390/electronics15122578) · [YOLO11m-Pose reference](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md) |

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

### Phụ lục kiểm toán Parameters từ các citation và model zoo — 28/10/2026

**Nguyên tắc:** Các ô đã được bổ sung trực tiếp trong Bảng 1, 2B và 3/3B khi tra được họ mô hình. **Không ghi Params backbone như Params full model**; [R] chỉ là mô hình tham chiếu, [D] là phép cộng các thành phần đã biết.

| Phương pháp | Citation/kiến trúc tìm được | Params tham chiếu (M) | Có thể xem là Params exact của hàng benchmark? |
|---|---|---:|---|
| Kausar 2026 | [Paper: Z-HandSegNet là U-Net với ResNet-34 encoder](https://doi.org/10.32604/cmc.2026.084514); [Torchvision ResNet-34](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34) | **21.797672 [R]** | **Không.** ResNet-34 bản classifier có FC; U-Net chỉ dùng encoder và còn decoder + Ef-RAFT + Transformer |
| Nayan 2024 | [Paper: BVLC GoogLeNet Caffe + BiLSTM](https://doi.org/10.1007/s00138-024-01565-9); [Torchvision GoogLeNet](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.googlenet.html) | **6.624904 [R] mỗi backbone** | **Không.** GoogLeNet Torchvision khác implementation Caffe; số streams, weight sharing, hidden size BiLSTM chưa khóa |
| Zhang 2026 | [Zhang Table 3](https://doi.org/10.3390/electronics15122578); [Ultralytics pose model zoo](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md) | **20.9 [R] pose + 0.08641 [P] head = 20.98641 [D/R]** | Có thể dùng **tổng tham chiếu của hai thành phần**, nhưng không coi là exact checkpoint/full pipeline được paper công bố. **YOLO11m-Pose khác YOLO11m detection 20.1M** |
| Xiong 2021 MD-GCN | [Table 1 paper gốc](https://doi.org/10.3390/electronics10050551) | **NR** (Model Size = **11.7 M**) | **Không.** Model Size không tự động là #Params. Giá trị ≈2.93M chỉ đúng **nếu** đó là 11.7 MB float32 thuần, chưa có bằng chứng nên không dùng |
| Xiong P3D reference | [Qiu P3D original Fig. 1](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf) | **≈65.3 [R]** (P3D-199) | **Không.** Fig. 1 ghi model size 261 MB; Xiong chưa ghi depth. [Bài 2022 khác](https://april.zju.edu.cn/wp-content/papercite-data/pdf/xv2022idg.pdf) báo P3D-199 ~66M, P3D-131 ~47M, P3D-63 ~25M |
| Xiong C3D reference | [MMAction2 16×112²](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md) | **78.4 [R]** | **Không.** Chưa xác nhận head/checkpoint Xiong |
| Xiong 3D ResNet reference | [Torchvision R3D-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html) | **33.371472 [R]** | **Không.** 3D ResNet có nhiều depth; bài Xiong chỉ dẫn Hara |
| RSC-GestureNet-Fusion | [Source RSCGestureNet](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/src/rsc_gesturenet/model.py) + [config hidden=96](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/configs/train.yaml) | **0.815270 [E]** | **Đúng phần fusion recognizer cấu hình code**, chưa gồm MediaPipe |
| RSC-GestureNet (Ours) | [RSC Table 1](https://arxiv.org/pdf/2608.02200), [weights index](https://github.com/chengli24/rsc-gesturenet-prcv2026/blob/main/WEIGHTS.md) | **≥0.815270 [E]** | **Chưa.** Full variant còn calibration/selection và các nhánh khác |
| TMMF / TD-Net / TCN+VC / ST-GCN+VC | Citation/paper chỉ xác nhận *họ kiến trúc*, chưa khóa cấu hình số tầng, head hoặc checkpoint | **NR** | **Không.** Không mượn Params từ mô hình mang tên gần giống trên dataset khác |

**Về GFLOPs:** 71.5 (YOLO11m-Pose @640, **một ảnh**), 40.70 (R3D-18 16×112²) và 38.5 (C3D 16×112²) là những số **[R] ở input/model zoo khác**. Không so hoặc cộng trực tiếp với VideoMamba 29.0261 GFLOPs/clip theo quy ước MAC của nhóm. Vẫn để NR cho GFLOPs tổng các pipeline chưa đủ mã nguồn.

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
