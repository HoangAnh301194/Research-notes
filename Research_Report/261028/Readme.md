# Báo cáo nghiên cứu ngày 28/10/2026

## A. Công việc đã làm.

- Đóng gói dữ liệu **EgoEv Gesture** lên Kaggle, kiểm tra cấu trúc/split và huấn luyện mô hình **VideoMamba-Tiny**.
- Xuất keyframe cho các trường hợp dự đoán **đúng/sai** từ hai mô hình VideoMamba đã huấn luyện trên **IPN Hand** và **CTP Gesture**, đồng thời tổng hợp các chỉ số đánh giá.

### 1. Đóng gói dataset EgoEv Gesture

**Cấu trúc và chia tập dữ liệu**

- Bộ dữ liệu sự kiện EgoEvGesture được đóng gói để chạy trên Kaggle; bài toán phân loại **38 lớp**.
- Ghi nhận trong quá trình kiểm tra dữ liệu gốc: **subject 5 và 10 trùng dữ liệu**, có **một tệp trùng giữa subject 4 và 8**. Đây là nguy cơ rò rỉ thông tin giữa các tập; không sử dụng các bản ghi trùng để đưa ra kết luận về khả năng tổng quát hóa.
- Notebook đang sử dụng **Heterogeneous Cleaned Split v1** với **train: 2.626 mẫu, subjects 3–7** và **validation: 489 mẫu, subject 8**; cả hai tập đều có 38 lớp. Việc đối chiếu số lượng, nhãn, đường dẫn và checksum CSV với manifest đã **PASS**. **Tập test chưa được đọc hoặc đánh giá trong notebook này.**
- Phạm vi kiểm tra tự động hiện thấy chủ yếu là train/validation; vẫn cần công bố bảng fingerprint/hash và xác nhận **không trùng nội dung giữa toàn bộ train–validation–test** trước khi dùng kết quả test trong bài báo.

**Cấu hình huấn luyện**

| Tham số | Giá trị theo notebook |
|---|---|
| Mô hình | VideoMamba-Tiny, 38 lớp; khoảng **6,965 triệu tham số** |
| Đầu vào | Sự kiện Nx4 → 16 khung, 3 kênh, 224×224 |
| Ba kênh | Số sự kiện dương, số sự kiện âm, tổng số sự kiện |
| Tiền xử lý | Chia 16 khoảng thời gian; log1p, chuẩn hóa percentile 99, ImageNet mean/std |
| Giới hạn mẫu sự kiện | 200.000 sự kiện/mẫu |
| Phần cứng | 2 × Tesla T4, PyTorch DataParallel (không phải DDP) |
| Global batch / tích lũy gradient | 2 / 4 → effective batch **8** |
| Optimizer | AdamW; learning rate ban đầu **1e-4**, weight decay **0,05** |
| Lịch học | Warmup 3 epochs, cosine schedule; min LR **1e-6** |
| Drop path / head dropout | 0,1 / 0,3 |
| Seed / Precision | 42 / FP32 (không AMP) |
| Kế hoạch | 60 epochs; mỗi phiên tối đa 8 epochs, có resume checkpoint |
| Mã nguồn | VideoMamba commit 37355c26d0ae99ca2459f6d4044a5f509031a79f |

**Kết quả huấn luyện sau 16 epochs**

| Mốc | Train Accuracy | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|
| Epoch 9 — checkpoint tốt nhất | 98,97% | **69,53%** | **65,26%** |
| Epoch 14 | 99,35% | 67,69% | 64,67% |
| Epoch 16 — checkpoint mới nhất | **99,31%** | **58,90%** | **53,01%** |

- Đã hoàn thành **16/60 epochs**. Kết quả tốt nhất được ghi tại **epoch 9**, không phải epoch 16.
- Kết thúc phiên: lưu **checkpoint-best.pth (epoch 9)**, **checkpoint-latest.pth (epoch 16)**, cấu hình và lịch sử train/validation. Checkpoint chứa trạng thái optimizer để tiếp tục.
- **Nhận định:** Train Accuracy gần 99% nhưng Validation Accuracy dao động và giảm còn 58,90% tại epoch 16 là **dấu hiệu quá khớp hoặc chênh lệch phân phối giữa subject train và validation**. Chưa đủ cơ sở xác định nguyên nhân duy nhất. Không được trình bày 69,53% là test Accuracy.
- Chuyển đổi sang PyTorch RMSNorm được dùng để khắc phục lỗi tương thích Triton trên Kaggle; có thể làm thay đổi số học của kernel. Khi so sánh phải ghi nhận phiên bản backend.

**Nguồn:** [Notebook EgoEvGesture và log train](./egoevgesture-videomamba-tiny%20%282%29.ipynb), các ô cấu hình, kiểm tra split, train và checkpoint (cells 4, 6, 18, 21, 23).


### 2. Xuất ra keyframe correct and error prediction để quan sát

Các hình keyframe được dùng để đối chiếu nhãn thực tế (GT) với nhãn dự đoán (Pred) và phân tích nguyên nhân nhận dạng/định vị thời gian chưa đúng. Kết quả isolated và continuous offline được báo cáo riêng do khác giao thức.

#### 2.1 IPN Hand

**Giao thức đánh giá**

- Dùng checkpoint đã fine-tune, **không train lại**. Mô hình nhận **16 RGB frames/clip, 224×224, 14 lớp**.
- **Isolated**: đánh giá **1.101 đoạn** từ tập test đã chú thích, loại nhãn D0X khỏi đánh giá isolated.
- **Continuous offline**: đánh giá **52 video, 212.574 frames**, cửa sổ **64 frames**, bước trượt **16 frames** (overlap 75%); lấy 16 khung làm đầu vào cho mỗi cửa sổ.
- Thử nghiệm early recognition bằng cách cắt một phần **đoạn đã biết biên ground truth** rồi lấy mẫu lại; **không phải đo thời gian phản hồi trực tuyến**.

**Kết quả định lượng**

| Chỉ số | Giá trị |
|---|---:|
| Isolated Top-1 Accuracy | **84,65%** |
| Isolated Macro-F1 | **78,34%** |
| Continuous RAW global Frame Accuracy | **83,73%** |
| Continuous RAW global Macro-F1 | **64,40%** |
| Continuous segment F1@10 / F1@25 / F1@50 | **79,45% / 77,58% / 69,91%** |
| Continuous mean edit score (MS-TCN convention) | **74,38%** |
| Mean temporal overlap (RAW, per-video) | **49,11%** |

**Độ chính xác nhận dạng khi chỉ quan sát một phần đoạn cử chỉ:**

| Tỷ lệ quan sát | 10% | 20% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Accuracy | 38,69% | 53,77% | 76,11% | 83,83% | 84,65% |

- Các lớp dễ nhầm hơn gồm **G02 (25,00%)**, **G01 (42,31%)** và **G08 (51,92%)**, dựa trên isolated per-class accuracy.
- Đã tạo hình gồm **đúng 16 khung được đưa vào mô hình**, kèm nhãn GT/Pred; isolated: **932 đúng, 169 sai**. Continuous: xuất **820 trường hợp ghép đúng, 706 trường hợp lỗi/không ghép được**, theo tiêu chí ghép nhãn và temporal IoU ≥ 0,5.
- **Giới hạn lưu trữ:** kết quả và hình IPN được ghi trong output Kaggle và notebook; **chưa xác minh có thư mục xuất hình IPN riêng trong đường dẫn báo cáo 261028 trên GitHub**. Muốn người xem duyệt tất cả case cần đồng bộ artifact này lên repository.

**Nguồn:** [Notebook IPN P1 — metrics và keyframe](./videomamba-ipn-hand-p1-metrics-keyframes-python313%20%281%29.ipynb), các cells 17, 19, 21, 23, 27, 29, 31, 33.


#### 2.2 CTP Gesture

**Giao thức đánh giá**

- Dùng checkpoint VideoMamba-Tiny **epoch 17**, **6.959.817 tham số**, **16 RGB frames/clip, 224×224**, **9 lớp**.
- Kiểm tra **11 video test chính thức**, **67.356 frames**, **1.728 GT segments** (trong đó 869 no-gesture và 859 gesture).
- Isolated: mỗi đoạn được gán nhãn là một clip. Continuous offline: cửa sổ **32 frames, stride 8**, overlap 75%, tổng **8.392 cửa sổ**; ghép xác suất dự đoán thành nhãn theo frame.
- Môi trường: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, Tesla T4, native Mamba CUDA; kiểm thử checkpoint/model forward đã **PASS**.

**Kết quả định lượng**

| Chỉ số | Giá trị |
|---|---:|
| Isolated Accuracy — 9 lớp | **98,55%** |
| Isolated Macro-F1 — 9 lớp | **97,96%** |
| Isolated gesture-only Accuracy — GT nhãn 1–8 | **98,02%** |
| Continuous RAW global Frame Accuracy | **83,98%** |
| Continuous RAW global Macro-F1 | **80,61%** |
| Continuous RAW mean per-video Frame Accuracy | **83,50%** |
| Continuous segment F1@10 / F1@25 / F1@50 | **96,05% / 94,75% / 85,55%** |
| Mean temporal overlap | **65,32%** |
| Mean gesture recall | **94,80%** |
| Mean matched onset / offset absolute error (tIoU ≥ 0,5) | **0,290 s / 0,446 s** |

**Early isolated Accuracy (9 lớp, dựa trên GT-prefix):**

| Tỷ lệ quan sát | 10% | 20% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Accuracy | 86,05% | 93,17% | 97,57% | 98,38% | 98,55% |

**Keyframe dự đoán đúng và sai**

- **Isolated:** xuất hình **1.728/1.728 clips**, gồm **1.703 đúng** và **25 sai**, không lỗi giải mã.
- **Continuous:** xuất **970/970 trường hợp**, gồm **725 đúng** và **245 lỗi**; mỗi hình chứa 16 khung đại diện từ cửa sổ mô hình cùng timeline GT/Pred. Đếm trường hợp continuous dựa trên ghép một-một cùng lớp với temporal IoU ≥ 0,5, **không phải 970 video độc lập**.
- Ví dụ isolated: [dự đoán đúng](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/isolated_16frame/correct/correct_00001.jpg) · [dự đoán sai](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/isolated_16frame/error/error_00002.jpg).
- Ví dụ continuous: [trường hợp đúng](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/continuous_16frame_timeline/correct/correct_00004.jpg) · [trường hợp lỗi](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/continuous_16frame_timeline/error/error_00001.jpg).
- Dữ liệu phục vụ tái kiểm chứng gồm classification report, isolated predictions, per-video CSV, early accuracy, phân tích biên đoạn và các bảng manifest từng trường hợp.

**Nguồn:** [Notebook CTP P1](./videomamba-ctpgesture-p1-metrics-keyframes-python3%20%283%29.ipynb), [thư mục kết quả CTP](./ctpResult/ctpgesture_videomamba_p1_evaluation/) và [kiểm toán P1](./ctpResult/ctpgesture_videomamba_p1_evaluation/p1_metric_audit/).


**Nhận xét tổng hợp về các kết quả keyframe và đánh giá**

- **CTP isolated có kết quả tốt nhưng continuous thấp hơn**, cho thấy phân loại đoạn đã tách sẵn chưa phản ánh đầy đủ độ chính xác nhận dạng trên video dài và biên bắt đầu/kết thúc cử chỉ.
- **IPN-Hand khó hơn CTP trong cấu hình hiện tại**, đặc biệt ở các lớp G01/G02/G08 và khả năng phân đoạn cử chỉ trên video dài. Đây là nhận xét trên hai tập, không phải kết luận mô hình nào vượt SOTA.
- **VideoMamba hiện đánh giá continuous theo cách offline** (cửa sổ chồng lấn và tổng hợp dự đoán); các metric này **không đồng nhất** với Online-F1, Early@10 hay Time-to-Correct-Command của hệ thống causal.
- Bảng đối chiếu thuật toán, Params/GFLOPs và phân loại giao thức đã có ở [báo cáo 22/10](../261022/Readme.md). Khi đưa vào bài báo cần phân biệt số đo từ checkpoint của nhóm, số tác giả công bố và số ước tính từ model zoo.


## B. Khó khăn

1. **EgoEv: khả năng tổng quát hóa theo subject chưa ổn định.** Validation tốt nhất ở epoch 9 nhưng giảm ở epoch 16; cần kiểm tra dữ liệu, phân phối lớp, subject và hiệu quả regularization; không nên chỉ tiếp tục train đủ 60 epochs theo kế hoạch mà không đánh giá.
2. **Nguy cơ trùng dữ liệu EgoEv:** phải hoàn thiện kiểm tra hash/nội dung trên cả ba tập train/validation/test, đặc biệt các subject 4/8 và 5/10.
3. **IPN: lỗi dự đoán và biên phân đoạn còn đáng kể.** Cần tổng hợp confusion matrix và một số trường hợp có tính đại diện, tránh chỉ chọn ảnh đẹp.
4. **Lưu trữ kết quả:** artifact hình ảnh IPN chưa được xác nhận hiện diện riêng trong folder GitHub; cần đồng bộ cùng CSV và metadata mô hình/checkpoint.
5. **So sánh SOTA:** các bài dùng split, modality và giao thức khác nhau; chưa thể tuyên bố phương pháp tốt hơn dựa vào các Accuracy không đồng nhất. Thời gian xử lý trên T4 không được gọi là latency của hệ thống causal thời gian thực.


## C. Công việc tiếp theo

- **P0 — Nghiệm thu dữ liệu EgoEv:** tạo bảng per-subject/per-class, thống kê số mẫu trước/sau loại trùng và đối chiếu hash toàn bộ train/validation/test. Khóa split trước khi đánh giá test.
- **P0 — Phân tích EgoEv epoch 9 so với epoch 16:** bảo toàn hai checkpoint, vẽ đường train/validation loss/accuracy/F1; xác định có cần thử early stopping, tăng regularization hoặc kiểm tra domain shift, không tự chọn tham số theo test.
- **P1 — Hoàn thiện keyframe IPN trên GitHub:** đưa các ảnh/manifest/CSV liên quan vào thư mục riêng, gắn nhãn true/pred, tên video, thời điểm bắt đầu/kết thúc và mẫu sai điển hình; lập bảng phân tích nguyên nhân G01/G02/G08.
- **P1 — Chốt bảng metrics thống nhất:** isolated Top-1/Macro-F1, Accuracy@10/20/50%, continuous Frame Accuracy/Macro-F1, Segment F1@10/25/50, Edit, lỗi biên; ghi rõ mẫu số, đơn vị và cách lấy trung bình.
- **P2 — Đối chứng bài báo:** rà lại IPN-Hand và CTPGesture v1 có dùng **RGB/video** (không chỉ skeleton), kiểm tra protocol/split; điền Params/GFLOPs từ paper gốc hoặc mã nguồn, giữ **NR** nếu chưa kiểm chứng.
- **P2 — Kế hoạch thử nghiệm tiếp:** sau khi khóa cấu hình, đánh giá EgoEv test đúng một quy trình cố định và so sánh với các baseline phù hợp; các thí nghiệm causal/online, latency hoặc hiệu năng thực tế cần một protocol riêng.

---

**Liên kết:** [Notebook EgoEv](./egoevgesture-videomamba-tiny%20%282%29.ipynb) · [Notebook IPN](./videomamba-ipn-hand-p1-metrics-keyframes-python313%20%281%29.ipynb) · [Notebook CTP](./videomamba-ctpgesture-p1-metrics-keyframes-python3%20%283%29.ipynb) · [CTP artifacts](./ctpResult/ctpgesture_videomamba_p1_evaluation/).

**Tính chất báo cáo:** số liệu nêu trên là kết quả đã ghi trong notebook; không có chạy lại huấn luyện hoặc suy luận trong lần biên tập README này.
