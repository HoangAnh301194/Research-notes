# Khảo sát tiến trình phát triển của Skeleton-based Gesture & Action Recognition

**Từ RNN/LSTM, TCN, GCN và Transformer đến Mamba/SSM — giai đoạn 2015–2026**

- **Ngày rà soát:** 02/09/2026.
- **Ngôn ngữ:** tiếng Việt; giữ nguyên tên bài báo để thuận tiện tra cứu.
- **Phạm vi chính:** nhận diện cử chỉ bàn tay và hành động con người từ chuỗi skeleton/keypoint.
- **Phạm vi mở rộng:** phân đoạn hành động theo thời gian trên chuỗi chưa cắt; liên hệ với IPN Hand và hệ thống dùng camera RGB.
- **Nội dung:** 26 công trình tiêu biểu, 4 tài liệu nền tảng/dữ liệu, giải thích kiến trúc, hướng đọc và đề xuất đánh giá.
- **Loại khảo sát:** tổng quan có chọn lọc và kiểm tra nguồn; không phải systematic review bảo đảm bao phủ toàn bộ công bố.

> **Đính chính quan trọng:** đã tìm thấy **GALFu-Mamba — Global And Local Fusion Mamba for Skeleton-based Temporal Action Segmentation**, thuộc IEEE Smart World Congress 2025. Vì vậy, nhận định trước đây rằng chưa thấy công trình Mamba cho skeleton temporal action segmentation cần được cập nhật. Không nên đặt tính mới của đề tài chỉ ở việc “đưa Mamba vào skeleton segmentation”. Xem [P25](#p25-galfu-mamba).

## Mục lục

1. [Phạm vi và cách đọc nguồn](#1-pham-vi)
2. [Bảng tiến trình phát triển](#2-tien-trinh)
3. [2015–2017: RNN/LSTM](#3-rnn-lstm)
4. [2017–2020: TCN và ST-GCN](#4-tcn-gcn)
5. [2019–2022: Adaptive GCN và graph đa thang](#5-adaptive-gcn)
6. [2020–2024: Attention và Transformer](#6-transformer)
7. [2024–2026: Mamba/SSM](#7-mamba)
8. [Nhánh chuỗi dài và temporal segmentation](#8-segmentation)
9. [GCN + Mamba hoạt động thế nào?](#9-gcn-mamba)
10. [Liên hệ IPN Hand và thiết kế thực nghiệm](#10-ipn-hand)
11. [Thứ tự đọc và sản phẩm cần rút ra](#11-doc-bai)
12. [Tài liệu bổ sung và giới hạn khảo sát](#12-bo-sung)

<a id="1-pham-vi"></a>

## 1. Phạm vi và cách đọc nguồn

### 1.1. Đây là lịch sử của nhánh skeleton-based

Bảng RNN → GCN → Transformer → Mamba đang xét **nhánh skeleton-based recognition**, không đại diện đầy đủ cho lịch sử action recognition trực tiếp trên RGB. Các hướng RGB như CNN hai luồng, I3D, TSN/TSM, VideoMAE và VideoMamba cần một khảo sát riêng.

Trong nhánh skeleton, đầu vào mô hình nhận diện thường là:

$$
X \in \mathbb{R}^{T\times J\times C},
$$

trong đó:

- \(T\): số bước thời gian đưa vào mạng.
- \(J\): số khớp/keypoint.
- \(C\): số kênh đặc trưng của mỗi khớp; ví dụ \(x,y\), hoặc \(x,y,z\), hoặc tọa độ kèm confidence.
- Batch và số người được lược bỏ để dễ trình bày.

**Ba kênh không tự động có nghĩa là tọa độ 3D:** một tensor có thể chứa \((x,y,\text{confidence})\). Phải đọc định nghĩa dữ liệu trước khi so sánh.

### 1.2. Phân biệt ba loại đầu ra

| Bài toán | Đầu vào | Đầu ra | Ý nghĩa |
|---|---|---|---|
| Isolated/trimmed classification | Một clip đã xác định đoạn hành động | Một nhãn \(y\) | Clip này là cử chỉ gì? |
| Temporal action/gesture segmentation | Chuỗi liên tục chưa cắt | Nhãn theo thời gian \(y_1,\ldots,y_T\) | Mỗi thời điểm đang làm gì? |
| Temporal action detection | Chuỗi chưa cắt | Các đoạn có thời điểm đầu/cuối và nhãn | Hành động nào xuất hiện, ở đoạn nào? |

Tên gọi giữa các paper không hoàn toàn thống nhất. Có paper gọi dense multi-label prediction là “detection”; hãy đối chiếu **nhãn, output và metric thực tế**, không chỉ tiêu đề. MS-Temba là ví dụ cần đọc theo cách này. [Nguồn MS-Temba][p26]

**Online/causal là một trục khác:** dự đoán tại \(t\) chỉ dùng dữ liệu đến \(t\). Mô hình offline có thể dùng các frame tương lai. Một mô hình xử lý video dài chưa chắc chạy online; một mô hình phân loại clip vẫn có thể nhận clip rất dài.

### 1.3. Quy ước liên kết và trạng thái xuất bản

- **PDF mở:** liên kết tới CVF/ECVA, arXiv, kho tác giả hoặc tạp chí mở. Bản arXiv có thể khác bản xuất bản cuối.
- **Trang bài:** dùng xác nhận tên, tác giả, venue và tìm bản cập nhật.
- **Chưa có PDF mở xác minh được:** giữ công trình vì liên quan, nhưng không hứa rằng có thể đọc toàn văn miễn phí.
- **Preprint:** đã xác minh bản arXiv; chưa xác nhận một venue phản biện tương ứng trong lần rà soát này. Không đồng nghĩa chắc chắn chưa từng được xuất bản.
- Ngày đăng arXiv, năm hội nghị, ngày xuất bản online và năm của tập kỷ yếu có thể khác nhau.

Một số trang CVF/nhà xuất bản chặn truy cập tự động. Tài liệu ưu tiên liên kết thay thế chính thống khi tìm được; **không khẳng định mọi URL đã được tải thành công trên máy người đọc**. Các mục thiếu bản mở được đánh dấu ngay tại chỗ.

### 1.4. Cách chọn bài

Ưu tiên các công trình tạo mốc về biểu diễn không gian, mô hình thời gian, học topology, attention hoặc selective SSM. Bổ sung bài hand gesture và segmentation để tránh chỉ nhìn vào NTU classification. Kiểm tra thông tin bằng trang hội nghị/tạp chí, arXiv, kho trường đại học và repository tác giả; nguồn thư mục thứ cấp chỉ hỗ trợ khi nguồn gốc không truy cập được.

Các nhóm truy vấn đã dùng gồm tên paper chính xác; “skeleton Mamba segmentation”; “Global And Local Fusion Mamba”; “skeleton temporal action segmentation”; và tên paper kèm “PDF”, “arXiv”.

<a id="2-tien-trinh"></a>

## 2. Bảng tiến trình phát triển

Các giai đoạn **chồng lấn**, không phải mô hình sau xóa bỏ hoàn toàn mô hình trước. Chẳng hạn, Res-TCN xuất hiện từ 2017, còn attention đã được dùng cùng TCN trước khi Transformer skeleton trở nên phổ biến.

| Giai đoạn | Biểu diễn/mô hình không gian | Mô hình thời gian | Công trình tiêu biểu | Câu hỏi thiết kế |
|---|---|---|---|---|
| 2015–2017 | Vector khớp, nhóm bộ phận, quy tắc duyệt | RNN/LSTM | [H-RNN][p01], [ST-LSTM][p02], [VA-LSTM][p03] | Làm sao học diễn biến tư thế, giảm ảnh hưởng góc nhìn và nhiễu? |
| 2017–2020 | Vector skeleton hoặc graph giải phẫu | TCN, graph + temporal convolution | [Res-TCN][p04], [ST-GCN][p05], [STA-Res-TCN][p06] | Có thể học chuyển động bằng convolution và giữ cấu trúc cơ thể không? |
| 2019–2022 | Graph học được, graph đa thang, topology theo kênh | Temporal convolution và toán tử graph không–thời gian | [2s-AGCN][p07], [MS-G3D][p08], [CTR-GCN][p09] | Bone graph cố định có đủ cho mọi hành động không? |
| 2020–2024 | Quan hệ khớp bằng attention, tuple, partition | Temporal/spatio-temporal attention | [DSTA-Net][p10], [STTFormer][p11], [SkateFormer][p12] | Học tương tác xa mà vẫn kiểm soát chi phí thế nào? |
| 2024–2026 | GCN, Transformer hoặc tuần tự hóa graph | Selective SSM/Mamba, thường kết hợp mô-đun khác | [Simba][p13], [SkelMamba][p14], [SkeMamba][p15], [ActionMamba][p16], [TSkel-Mamba][p17] | Đặt SSM ở đâu và scan skeleton theo thứ tự nào? |
| 2026 | GCN/GNN gọn, topology thích nghi | Temporal Mamba | [GCN-Mamba][p18], [SkeletonMamba][p19], [ME-GCN][p20] | Tối ưu cân bằng accuracy, tham số và độ trễ ra sao? |
| Nhánh song song 2022–2026 | Skeleton encoder cho chuỗi chưa cắt | Dilated TCN, attention, Mamba | [MS-GCN][p21], [LAC][p22], [ME-ST][p24], [GALFu-Mamba][p25] | Dự đoán liên tục và bảo toàn biên hành động thế nào? |

**Diễn giải:** tiến trình không chỉ là thay “bộ xử lý chuỗi”. Các bước phát triển còn thay đổi cách biểu diễn khớp, cách học quan hệ, supervision, dữ liệu huấn luyện và đầu ra bài toán. [Nguồn: P01–P26 trong các mục bên dưới.]

<a id="3-rnn-lstm"></a>

## 3. Giai đoạn 2015–2017: RNN/LSTM

Trực giác của giai đoạn này là xem hành động như diễn biến của tư thế. Một trạng thái nhớ được cập nhật khi đọc các khớp/frame, thay vì phân loại từng ảnh độc lập.

<a id="p01-h-rnn"></a>

### P01. H-RNN — CVPR 2015

**Tên:** *Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition*  
**Tác giả:** Yong Du, Wei Wang, Liang Wang.  
**Đọc:** [PDF mở][p01-pdf] · [Trang CVPR][p01].

- Chia cơ thể thành năm nhóm và học bằng các nhánh recurrent; đặc trưng được hợp nhất dần theo cấp.
- **Spatial:** cấu trúc nhóm bộ phận thiết kế trước. **Temporal:** bidirectional RNN.
- **Vai trò:** mốc tiêu biểu của việc đưa hiểu biết cấu trúc cơ thể vào mạng chuỗi.
- **Lưu ý khi tái sử dụng:** nhánh hai chiều dùng thông tin tương lai; cần sửa nếu yêu cầu causal.

**Câu hỏi khi đọc:** vì sao hợp nhất theo bộ phận có thể tốt hơn ghép tất cả tọa độ ngay từ đầu? [Nguồn][p01]

<a id="p02-st-lstm"></a>

### P02. ST-LSTM với Trust Gates — ECCV 2016

**Tên:** *Spatio-Temporal LSTM with Trust Gates for 3D Human Action Recognition*  
**Tác giả:** Jun Liu, Amir Shahroudy, Dong Xu, Gang Wang.  
**Đọc:** [PDF arXiv][p02-pdf] · [Trang bài][p02].

- Mở rộng LSTM để mô hình hóa cả không gian và thời gian, dùng cách duyệt skeleton theo cây.
- Trust gate điều chỉnh ảnh hưởng của đầu vào không đáng tin cậy lên bộ nhớ.
- **Vai trò:** cho thấy xử lý nhiễu/che khuất đã là vấn đề từ rất sớm.
- **Giới hạn cần xem:** ảnh hưởng của thứ tự duyệt và giả định về độ tin cậy skeleton.

**Liên hệ:** khi trích hand keypoint từ RGB, “khớp bị mất” và “khớp đo sai” cũng cần được mô hình hóa. [Nguồn][p02]

<a id="p03-va-lstm"></a>

### P03. VA-LSTM — ICCV 2017

**Tên:** *View Adaptive Recurrent Neural Networks for High Performance Human Action Recognition From Skeleton Data*  
**Tác giả:** Pengfei Zhang và cộng sự.  
**Đọc:** [PDF arXiv][p03-pdf] · [Trang ICCV][p03].

- Học biến đổi góc nhìn skeleton trước/trong quá trình nhận diện bằng LSTM.
- Mục đích là giảm sai khác do camera trong khi vẫn giữ diễn biến chuyển động.
- **Vai trò:** đưa bước chuẩn hóa hình học vào mô hình học được.
- **Giới hạn:** chuẩn hóa góc nhìn không thay thế mô hình tương tác khớp; hiệu quả còn phụ thuộc chất lượng tọa độ.

**Câu hỏi khi đọc:** biến đổi nào cần bất biến, và biến đổi nào đang mang thông tin phân biệt hành động? [Nguồn][p03]

<a id="4-tcn-gcn"></a>

## 4. Giai đoạn 2017–2020: TCN và ST-GCN

Temporal convolution học các mẫu chuyển động trong cửa sổ thời gian. Nhiều tầng hoặc dilation mở rộng vùng nhìn. GCN bổ sung cấu trúc: đặc trưng của một khớp được kết hợp với những khớp có quan hệ trên graph.

<a id="p04-res-tcn"></a>

### P04. Res-TCN — CVPR Workshops 2017

**Tên:** *Interpretable 3D Human Action Analysis with Temporal Convolutional Networks*  
**Tác giả:** Tae Soo Kim, Austin Reiter.  
**Đọc:** [PDF CVF][p04-pdf] · [Trang hội thảo][p04] · [arXiv](https://arxiv.org/abs/1704.04516).

- Dùng residual temporal convolution cho skeleton action recognition.
- Nhấn mạnh khả năng phân tích các đặc trưng không–thời gian học được.
- **Benchmark tiêu biểu:** NTU RGB+D.
- **Vai trò:** một baseline TCN phù hợp để kiểm tra xem temporal model phức tạp hơn có thực sự cần thiết.

**Lưu ý thư mục:** đây là **CVPR Workshops**, không ghi thành paper thuộc main conference. [Nguồn][p04]

<a id="p05-st-gcn"></a>

### P05. ST-GCN — AAAI 2018

**Tên:** *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*  
**Tác giả:** Sijie Yan, Yuanjun Xiong, Dahua Lin.  
**Đọc:** [PDF arXiv][p05-pdf] · [Trang AAAI][p05].

- Biểu diễn skeleton bằng graph: cạnh trong frame thể hiện quan hệ cơ thể; chiều thời gian nối diễn biến khớp.
- **Spatial:** graph convolution. **Temporal:** convolution theo thời gian trong kiến trúc triển khai.
- **Benchmark:** NTU RGB+D và Kinetics-Skeleton.
- **Vai trò:** baseline nền tảng để hiểu các dòng GCN skeleton về sau.
- **Giới hạn:** topology giải phẫu đặt trước có thể chưa thể hiện trực tiếp tương tác đặc thù hành động.

**Tránh nhầm:** khác bài *Spatio-Temporal Graph Convolution for Skeleton Based Action Recognition* của Chaolong Li và cộng sự, cũng năm 2018. [Nguồn][p05]

<a id="p06-sta-res-tcn"></a>

### P06. STA-Res-TCN — ECCV Workshops 2018

**Tên:** *Spatial-Temporal Attention Res-TCN for Skeleton-Based Dynamic Hand Gesture Recognition*  
**Tác giả:** Jingxuan Hou và cộng sự.  
**Đọc:** [PDF từ Tsinghua][p06-pdf] · [DOI kỷ yếu][p06].

- Ghép tọa độ khớp thành đặc trưng từng frame; dùng Res-TCN và nhánh attention mask.
- Attention tăng trọng số các đặc trưng/thời điểm hữu ích.
- **Benchmark:** DHG-14/28 và SHREC’17.
- **Vai trò:** mốc liên quan trực tiếp đến dynamic hand gesture.
- **Giới hạn:** benchmark phân loại cử chỉ không tự chứng minh chất lượng phát hiện biên trong stream.

**Lưu ý năm:** hội thảo thuộc ECCV 2018; bản kỷ yếu có thể được ghi năm 2019. [Nguồn PDF][p06-pdf]

<a id="5-adaptive-gcn"></a>

## 5. Giai đoạn 2019–2022: Adaptive GCN và graph đa thang

Hai bàn tay có thể tương tác mạnh khi vỗ tay dù không nối trực tiếp bằng một xương. Vì vậy, quan hệ hữu ích cho nhận diện không chỉ là quan hệ giải phẫu. Nhóm này mở rộng hoặc học topology từ dữ liệu.

<a id="p07-2s-agcn"></a>

### P07. 2s-AGCN — CVPR 2019

**Tên:** *Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition*  
**Tác giả:** Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu.  
**Đọc:** [PDF CVF][p07-pdf] · [Trang CVPR][p07].

- Học topology thay vì giữ nguyên cho mọi tầng/mẫu.
- Hai luồng xử lý joint và bone, bổ sung thông tin vị trí với quan hệ hình học giữa khớp.
- **Benchmark:** NTU RGB+D, Kinetics-Skeleton.
- **Vai trò:** mốc của adaptive graph và khai thác nhiều biểu diễn từ cùng skeleton.
- **Khi so sánh:** ghi rõ single-stream hay fusion; không so accuracy hai luồng với một luồng mà bỏ qua chi phí. [Nguồn][p07]

<a id="p08-ms-g3d"></a>

### P08. MS-G3D — CVPR 2020

**Tên:** *Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition*  
**Tác giả:** Ziyu Liu và cộng sự.  
**Đọc:** [PDF CVF][p08-pdf] · [Trang CVPR][p08] · [arXiv](https://arxiv.org/abs/2003.14111).

- Tách đóng góp các lân cận graph ở nhiều khoảng cách.
- Toán tử G3D cho phép truyền thông tin trực tiếp qua quan hệ không–thời gian.
- **Benchmark:** NTU60, NTU120, Kinetics Skeleton 400.
- **Vai trò:** mở rộng tương tác vượt ngoài khớp lân cận trong một frame.
- **Giới hạn khi triển khai:** cần kiểm tra chi phí của graph đa thang trên độ dài chuỗi mục tiêu. [Nguồn][p08]

<a id="p09-ctr-gcn"></a>

### P09. CTR-GCN — ICCV 2021

**Tên:** *Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition*  
**Tác giả:** Yuxin Chen và cộng sự.  
**Đọc:** [PDF CVF][p09-pdf] · [Trang ICCV][p09].

- Học một topology chung rồi tinh chỉnh bằng tương quan riêng cho từng kênh.
- Kết hợp graph refinement với temporal modeling để tạo CTR-GCN.
- **Benchmark:** NTU60, NTU120, NW-UCLA.
- **Vai trò:** baseline mạnh để đánh giá một spatial encoder mới.
- **Câu hỏi khi đọc:** có cần dùng cùng quan hệ khớp cho mọi kênh đặc trưng không?

**Lưu ý:** “GCN” mô tả thành phần graph; một mạng mang tên GCN vẫn có thể dùng temporal convolution đáng kể. [Nguồn][p09]

<a id="6-transformer"></a>

## 6. Giai đoạn 2020–2024: Attention và Transformer

Attention cho phép học mức liên quan giữa các phần tử mà không giới hạn vào cạnh giải phẫu trực tiếp. Đổi lại, việc cho mọi khớp ở mọi frame tương tác đầy đủ có thể tốn kém; cách tách trục, tạo tuple hoặc chia partition trở thành phần quan trọng của thiết kế.

<a id="p10-dsta-net"></a>

### P10. DSTA-Net — ACCV 2020

**Tên:** *Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition*  
**Tác giả:** Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu.  
**Đọc:** [PDF CVF][p10-pdf] · [Trang ACCV][p10] · [PDF arXiv](https://arxiv.org/pdf/2007.03263) · [Code](https://github.com/lshiwjx/DSTA-Net).

- Tách attention không gian và thời gian; bổ sung positional encoding và regularization phù hợp skeleton.
- **Benchmark:** SHREC, DHG, NTU60, NTU120.
- **Vai trò:** cầu nối rõ giữa hand gesture và full-body action trong cùng một hướng kiến trúc.
- **Cần kiểm tra:** số luồng và cách fusion khi tái lập kết quả.

**Lưu ý tên:** bản arXiv dùng tiêu đề ngắn hơn, không có “Action-Gesture”. [Nguồn][p10]

<a id="p11-sttformer"></a>

### P11. STTFormer — preprint 2022

**Tên:** *Spatio-Temporal Tuples Transformer for Skeleton-Based Action Recognition*  
**Tác giả:** Helei Qiu, Biao Hou, Bo Ren, Xiaohua Zhang.  
**Đọc:** [PDF arXiv][p11-pdf] · [Trang bài][p11] · [Code](https://github.com/heleiqiu/STTFormer).

- Nhóm thông tin của nhiều khớp trong các frame liên tiếp thành tuple.
- Học quan hệ trong tuple và tổng hợp thông tin giữa các frame xa hơn.
- **Benchmark:** NTU RGB+D 60/120 theo repository.
- **Vai trò:** cho thấy cách tạo token ảnh hưởng trực tiếp đến loại quan hệ học được.
- **Trạng thái:** nguồn arXiv và citation của repository ghi 2022; lần rà soát này chưa xác nhận venue khác.

**Câu hỏi khi đọc:** tuple bao nhiêu frame là đủ cho chuyển động ngắn mà không làm mất chi tiết? [Nguồn][p11]

<a id="p12-skateformer"></a>

### P12. SkateFormer — ECCV 2024

**Tên:** *SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition*  
**Tác giả:** Jeonghyeok Do, Munchurl Kim.  
**Đọc:** [PDF ECVA][p12-pdf] · [Trang ECCV][p12].

- Chia quan hệ theo tổ hợp: khớp gần/xa và frame gần/xa.
- Attention hoạt động trong các partition, giảm nhu cầu attention toàn cục trên toàn bộ token.
- **Vai trò:** đối chứng thích hợp cho tuyên bố Mamba hiệu quả hơn Transformer.
- **Giới hạn diễn giải:** không được lấy chi phí của full attention ngây thơ làm đại diện cho mọi Transformer.

**Câu hỏi khi đọc:** partition giữ được những tương tác nào, và bỏ bớt những tương tác nào? [Nguồn][p12]

<a id="7-mamba"></a>

## 7. Giai đoạn 2024–2026: Mamba/SSM

Mamba dùng trạng thái để truyền thông tin theo chuỗi, với cơ chế chọn lọc phụ thuộc đầu vào. Khi áp dụng vào skeleton, hai quyết định quan trọng là **phần nào dùng graph/attention** và **SSM scan theo chiều nào**. [Mamba gốc][r02]

Không có một kiến trúc duy nhất mang nghĩa “GCN + Mamba”. Mamba có thể nằm giữa các graph block, theo thời gian của từng khớp, sau pooling theo frame, hoặc trên chuỗi token không–thời gian.

<a id="p13-simba"></a>

### P13. Simba — preprint 2024

**Tên:** *Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos*  
**Tác giả:** Soumyabrata Chaudhuri, Saumik Bhattacharya.  
**Đọc:** [PDF arXiv][p13-pdf] · [Trang bài][p13].

- U-ShiftGCN trích spatial features; Mamba ở giữa xử lý thời gian; ShiftTCN tiếp tục tinh chỉnh temporal features.
- **Benchmark:** NTU60, NTU120, NW-UCLA.
- **Vai trò:** một công trình sớm, dễ thấy cách kết hợp GCN và Mamba.
- **Điểm cần đọc:** ablation U-ShiftGCN không có Mamba, vì spatial backbone tự nó đã cải thiện baseline.
- **Trạng thái:** xác minh preprint tháng 4/2024; không tự gán venue.

**Tránh suy diễn:** toàn bộ cải thiện của hệ lai chưa chắc đến từ Mamba. [Nguồn][p13]

<a id="p14-skelmamba"></a>

### P14. SkelMamba — preprint 2024

**Tên:** *SkelMamba: A State Space Model for Efficient Skeleton Action Recognition of Neurological Disorders*  
**Tác giả:** Niki Martinel, Mariano Serrao, Christian Micheloni.  
**Đọc:** [PDF arXiv][p14-pdf] · [Trang bài][p14].

- Phân chia xử lý thành spatial, temporal và spatio-temporal streams.
- Scan nhiều hướng, có định hướng theo cấu trúc cơ thể.
- **Benchmark:** NTU60, NTU120, NW-UCLA; có thêm dữ liệu ứng dụng lâm sàng.
- **Vai trò:** ví dụ SSM dùng rộng hơn việc thay temporal module.
- **Giới hạn:** kết quả classification không thay thế đánh giá segmentation/online.

**Tránh nhầm tên:** SkelMamba khác SkeMamba và SkeletonMamba. [Nguồn][p14]

<a id="p15-skemamba"></a>

### P15. SkeMamba — The Visual Computer 2025

**Tên:** *Dual-path spatio-temporal Mamba for skeleton-based action recognition*  
**Tác giả:** Jie Zhao, Ju Dai, Feng Zhou, Junjun Pan, Hongwen Xu.  
**Xuất bản:** The Visual Computer 41, 6507–6519; DOI 10.1007/s00371-025-03950-5.  
**Đọc:** [Trang nhà xuất bản][p15] · [Hồ sơ trường Beihang](https://research.buaa.edu.cn/en/publications/dual-path-spatio-temporal-mamba-for-skeleton-based-action-recogni/).

- Adaptive Topology Transformation chuyển skeleton graph thành biểu diễn tuần tự.
- Dùng dual-path spatial–temporal Mamba và gating cho đặc trưng chuyển động.
- **Benchmark:** NTU và NW-UCLA theo abstract.
- **Vai trò:** nghiên cứu tác động của tuần tự hóa graph trước khi scan.
- **Toàn văn:** chưa xác minh được PDF mở ổn định; phân tích ở mức abstract/metadata, không tái dựng chi tiết thuật toán. [Nguồn][p15]

<a id="p16-actionmamba"></a>

### P16. ActionMamba — Electronics 2025

**Tên:** *ActionMamba: Action Spatial–Temporal Aggregation Network Based on Mamba and GCN for Skeleton-Based Action Recognition*  
**Tác giả:** Jinglong Wen, Dan Liu, Bin Zheng.  
**Xuất bản:** Electronics 14(18), 3610; 11/09/2025.  
**Đọc:** [Toàn văn HTML mở][p16] · [PDF nhà xuất bản][p16-pdf].

- Kết hợp GCN và Mamba để khai thác tương tác không–thời gian.
- Nhắm tới hạn chế của quan hệ cục bộ và tương tác thời gian giữa các khớp khác nhau.
- **Vai trò:** thêm bằng chứng rằng “GCN + Mamba” đã là hướng có công trình công bố.
- **Lưu ý:** website mở nhưng truy cập tự động có thể bị giới hạn; dùng HTML nếu PDF không tải được.

**Cần kiểm tra trước tái lập:** cấu hình luồng, tensor scan và đóng góp riêng của từng module. [Nguồn][p16]

<a id="p17-tskel-mamba"></a>

### P17. TSkel-Mamba — preprint 2025

**Tên:** *TSkel-Mamba: Temporal Dynamic Modeling via State Space Model for Human Skeleton-based Action Recognition*  
**Tác giả:** Yanan Liu và cộng sự.  
**Đọc:** [PDF arXiv][p17-pdf] · [Trang bài][p17].

- **Spatial:** Transformer. **Temporal:** Mamba trong Temporal Dynamic Modeling block.
- Multi-scale Temporal Interaction dùng Cycle operators để tăng tương tác kênh–thời gian.
- **Benchmark:** NTU60, NTU120, NW-UCLA, UAV-Human.
- **Vai trò:** ví dụ phân công spatial/temporal rõ ràng, phù hợp để đối chiếu với GCN + Mamba.
- **Trạng thái:** bản arXiv ngày 12/12/2025; chưa xác minh venue phản biện khác.

**Câu hỏi khi đọc:** lợi ích đến từ SSM hay mô-đun tương tác bổ sung? [Nguồn][p17]

<a id="p18-gcn-mamba"></a>

### P18. GCN-Mamba — JAIT 2026

**Tên:** *GCN-Mamba: A Semantic-guided Graph Convolutional Network with Mamba State Space Models for Skeleton-based Action Recognition*  
**Tác giả:** Amine Mansouri, Abdellah Elzaar, Toufik Bakir, Smain Femmam.  
**Xuất bản:** JAIT 17(7), 1269–1277; 10/07/2026.  
**Đọc:** [PDF mở][p18-pdf] · [Trang tạp chí][p18].

- Adaptive adjacency GCN cho không gian, Mamba cho thời gian.
- Mục tiêu chính là so với **ImpSGN**, mô hình trước của nhóm tác giả.
- Báo cáo giảm khoảng 72% tham số, từ 4,0M xuống 1,1M.
- **Vai trò:** ví dụ đánh đổi độ chính xác và độ phức tạp.
- **Giới hạn:** mức giảm 72% chỉ có ý nghĩa với đối chứng đó; không đại diện cho mọi GCN/Transformer. [Nguồn][p18]

<a id="p19-skeletonmamba"></a>

### P19. SkeletonMamba — ICPR 2026

**Tên:** *SkeletonMamba: A Lightweight Mamba-Based Architecture for Action Recognition*  
**Tác giả:** Sanzhar Abdrakhim, Luca Rossi.  
**Đọc:** [Trang Springer][p19] · [Kho PolyU](https://ira.lib.polyu.edu.hk/handle/10397/120542) · [Code tác giả](https://github.com/Spanchsan/URIS_HarmAssessment).

- Kết hợp lightweight GNN và Temporal Mamba; repository mô tả bỏ các TCN dư thừa.
- **Benchmark:** NTU60 và NTU120.
- **Vai trò:** đánh giá hiệu quả của một temporal backbone tập trung vào Mamba.
- **PDF:** kho PolyU ghi embargo đến 03/08/2027; chưa tìm được bản mở thay thế.
- **Năm thư mục:** Springer ghi online 03/08/2026, hội nghị ICPR 2026, nhưng mục “Cite this paper” dùng 2027. Khi xuất BibTeX cần giữ metadata của bản đang trích.

**Số liệu repository:** được ghi là preliminary; cần kiểm tra lại toàn văn trước đưa vào bảng xếp hạng. [Nguồn Springer][p19] · [Nguồn code](https://github.com/Spanchsan/URIS_HarmAssessment)

<a id="p20-me-gcn"></a>

### P20. ME-GCN — Computer Animation and Virtual Worlds 2026

**Tên:** *Mamba-Enhanced Graph Convolutional Network for Skeleton-Based Stereoscopic Hand Gesture Recognition in Augmented Reality*  
**Tác giả:** Fucheng Wan, Jian Teng.  
**Xuất bản:** 14/07/2026; DOI 10.1002/cav.70164.  
**Đọc:** [Trang bài/abstract][p20].

- Kết hợp Spatial Graph Mamba, topology refinement dùng tín hiệu binocular depth, và fusion nhiều loại đặc trưng.
- **Benchmark:** SHREC’17, DHG-14/28, AR-Stereo-Gesture.
- **Vai trò:** liên quan trực tiếp đến hand skeleton + Mamba.
- **Giới hạn chuyển giao:** cơ chế dùng độ sâu stereo không mặc nhiên phù hợp hệ chỉ có RGB đơn mắt.
- **Trạng thái PDF:** đường dẫn PDF chuyển về abstract khi kiểm tra; chưa xác minh bản mở.
- Abstract ghi continuous streaming gesture recognition là hướng mở rộng tương lai. [Nguồn][p20]

<a id="8-segmentation"></a>

## 8. Nhánh chuỗi dài và temporal segmentation

Đây là phần cần đọc để chuyển từ phân loại clip sang hiểu video liên tục. Không chỉ đổi classification head: dữ liệu nền, biên chuyển động, nhãn theo frame, sự chồng lấn hành động và điều kiện causal đều có thể thay đổi bài toán.

<a id="p21-ms-gcn"></a>

### P21. MS-GCN — preprint 2022; có bản IEEE TETC

**Tên:** *Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks*  
**Tác giả:** Benjamin Filtjens, Bart Vanrumste, Peter Slaets.  
**Đọc:** [PDF arXiv][p21-pdf] · [Trang bài][p21] · [DOI IEEE](https://doi.org/10.1109/TETC.2022.3230912).

- Dùng spatial graph convolution và dilated temporal convolution ở giai đoạn dự đoán đầu.
- Các giai đoạn tiếp theo tinh chỉnh kết quả theo thời gian.
- **Vai trò:** baseline trực tiếp cho skeleton action segmentation.
- Bài đánh giá trên năm tác vụ; cần giữ đúng split/protocol thay vì chỉ so nhãn dataset.

**Câu hỏi khi đọc:** spatial modeling cải thiện gì so với chỉ dùng temporal convolution trên vector skeleton? [Nguồn][p21]

<a id="p22-lac"></a>

### P22. LAC — ICCV 2023

**Tên:** *LAC – Latent Action Composition for Skeleton-based Action Segmentation*  
**Tác giả:** Di Yang và cộng sự.  
**Đọc:** [PDF CVF][p22-pdf] · [Trang ICCV][p22] · [Code](https://github.com/walker1126/Latent_Action_Composition).

- Tổng hợp chuyển động phức hợp trong latent space.
- Dùng contrastive learning để học biểu diễn skeleton phục vụ segmentation.
- **Benchmark:** TSU, Charades, PKU-MMD.
- **Vai trò:** nhắc rằng tiến bộ còn đến từ pretraining và biểu diễn, không chỉ temporal backbone.
- **Lưu ý:** các tác vụ có hành động chồng lấn cần kiểm tra cách mã hóa multi-label.

**Tránh diễn giải sai:** LAC không đơn thuần là “GCN + TCN/Transformer”; đóng góp trung tâm là action composition và học biểu diễn. [Nguồn][p22]

<a id="p23-motion-aware"></a>

### P23. Motion-aware and Temporal-enhanced ST-GCN — Neurocomputing 2024

**Tên:** *A motion-aware and temporal-enhanced Spatial–Temporal Graph Convolutional Network for skeleton-based human action segmentation*  
**Tác giả:** Shurong Chai và cộng sự.  
**Xuất bản:** Neurocomputing 580, 127482.  
**Đọc:** [Trang bài][p23] · [Hồ sơ trường](https://pure.fujita-hu.ac.jp/en/publications/a-motion-aware-and-temporal-enhanced-spatialtemporal-graph-convol/) · [Code](https://github.com/11yxk/openpack).

- Kết hợp motion-aware module, temporal convolution đa thang, temporal-enhanced GCN và refinement.
- **Vai trò:** đối chứng cho giả thuyết cần đồng thời giữ chuyển động cục bộ và ngữ cảnh dài.
- **PDF:** chưa xác minh được bản mở; mô tả dựa trên abstract và hồ sơ xuất bản.

**Câu hỏi khi đọc:** cải thiện biên đến từ motion input, temporal module hay refinement? [Nguồn][p23]

<a id="p24-me-st"></a>

### P24. ME-ST — IEEE TNNLS 2025

**Tên:** *Snippet-Aware Transformer With Multiple Action Elements for Skeleton-Based Action Segmentation*  
**Tác giả:** Haoyu Ji và cộng sự.  
**Xuất bản:** IEEE TNNLS 36(9), 17462–17476.  
**Đọc:** [PubMed/abstract][p24] · [DOI](https://doi.org/10.1109/TNNLS.2025.3563025) · [Code](https://github.com/HaoyuJi/ME-ST).

- Attention theo joint, frame và scale trong các snippet.
- Nhằm nhận ra bộ phận và các pha chuyển động có tính phân biệt.
- **Code:** có train/test, nhánh boundary refinement, pretrained models và hướng dẫn dữ liệu.
- **Protocol trong repo:** LARa, HuGaDB, TCG, PKU-MMD X-sub và X-view; hai mục cuối là hai protocol của cùng dataset.
- **PDF:** chưa tìm được bản mở ngoài IEEE.

**Vai trò:** baseline Transformer liên quan trực tiếp đến segmentation. [Nguồn][p24] · [Repo](https://github.com/HaoyuJi/ME-ST)

<a id="p25-galfu-mamba"></a>

### P25. GALFu-Mamba — IEEE Smart World Congress 2025

**Tên:** *Global And Local Fusion Mamba for Skeleton-based Temporal Action Segmentation*  
**Tác giả:** Shuaibiao Zhang, Tao Zhu, Zhaoping Liao, Liming Chen.  
**Xuất bản:** IEEE SWC 2025, trang 1064–1071; DOI 10.1109/SWC65939.2025.00171.  
**Đọc:** [IEEE Xplore][p25] · [Hồ sơ thư mục](https://www.researchgate.net/publication/401008153_Global_And_Local_Fusion_Mamba_for_Skeleton-based_Temporal_Action_Segmentation).

- Đây là công trình trực tiếp về **Mamba + skeleton temporal segmentation**.
- Abstract được nguồn thư mục lập chỉ mục mô tả phân chia chuỗi, học đặc trưng local/global và fusion thích nghi.
- **PDF:** đã tìm đường dẫn IEEE nhưng không truy cập được; chưa tìm được bản mở từ tác giả/arXiv.
- **Mức kiểm chứng:** đã đối chiếu tiêu đề/venue/DOI từ nguồn lập chỉ mục; chưa đọc toàn văn, không báo lại số liệu hay khẳng định causal.

**Hệ quả:** không thể coi toàn bộ “Mamba cho skeleton segmentation” là khoảng trống chưa có nghiên cứu. [Nguồn IEEE][p25] · [Abstract được lập chỉ mục](https://eurekamag.com/research/105/648/105648665.php)

<a id="p26-ms-temba"></a>

### P26. MS-Temba — CVPR 2026; preprint từ 2025

**Tên bản hội nghị:** *MS-Temba: Multi-Scale Temporal Mamba for Understanding Long Untrimmed Videos*  
**Tác giả:** Arkaprava Sinha và cộng sự.  
**Đọc:** [PDF arXiv][p26-pdf] · [Trang CVPR][p26] · [Project](https://mstemba.github.io/) · [Code](https://github.com/thearkaprava/MS-Temba).

- Temporal Mamba nhiều thang, dilated SSM và mô-đun hợp nhất đặc trưng.
- **Dữ liệu vào temporal model:** video features; repository hướng dẫn I3D và CLIP.
- **Vai trò:** tham khảo thiết kế mô hình cho video chưa cắt và biên thời gian.
- **Giới hạn:** không phải skeleton-based model; cần ghi riêng trong bảng.
- **Phiên bản:** preprint ban đầu có tiêu đề khác; tránh trộn số tham số và kết quả giữa các phiên bản.

**Câu hỏi khi đọc:** khi đưa SSM vào dense prediction, cách giữ chi tiết ngắn hạn thay đổi thế nào? [Nguồn][p26] · [Repo](https://github.com/thearkaprava/MS-Temba)

<a id="9-gcn-mamba"></a>

## 9. GCN + Mamba hoạt động thế nào?

Phần này là **giải thích khái niệm và một thiết kế minh họa**, không phải kiến trúc chung của tất cả paper phía trên.

### 9.1. Skeleton vừa là graph, vừa là chuỗi

Trong một frame, các khớp có quan hệ không gian. Qua nhiều frame, tư thế và vị trí thay đổi. Vì vậy có thể phân công:

| Thành phần | Công việc |
|---|---|
| Spatial GCN | Kết hợp thông tin các khớp theo graph |
| Pooling/projection | Tạo đặc trưng frame hoặc nhóm bộ phận |
| Temporal Mamba | Học diễn biến của đặc trưng qua thời gian |
| Prediction head | Trả nhãn clip, nhãn frame hoặc biên |

Đây là lựa chọn thiết kế. ST-GCN và MS-G3D có cả xử lý thời gian; SkelMamba có nhiều loại stream. Không nên định nghĩa “GCN luôn chỉ spatial” hoặc “Mamba luôn chỉ temporal”.

### 9.2. Một lớp GCN cơ bản

Trực giác: mỗi khớp nhận thông tin từ hàng xóm và chính nó. Ví dụ khuỷu tay tổng hợp thông tin từ vai, khuỷu và cổ tay.

Một dạng GCN chuẩn là:

$$
H^{(\ell+1)}
=
\sigma\left(
\widetilde D^{-1/2}
\widetilde A
\widetilde D^{-1/2}
H^{(\ell)}W^{(\ell)}
\right).
$$

Trong đó:

- \(A\in\mathbb{R}^{J\times J}\): adjacency.
- \(\widetilde A=A+I\): thêm self-loop.
- \(\widetilde D_{ii}=\sum_j\widetilde A_{ij}\): degree **của graph đã thêm self-loop**.
- \(H^{(\ell)}\in\mathbb{R}^{J\times d_\ell}\): đặc trưng khớp tại tầng \(\ell\).
- \(W^{(\ell)}\in\mathbb{R}^{d_\ell\times d_{\ell+1}}\): trọng số học được.
- \(\sigma\): hàm phi tuyến.

**Sửa điểm dễ nhầm:** khi công thức dùng \(\widetilde A\), ma trận degree đi cùng phải tính từ \(\widetilde A\), không phải từ \(A\) ban đầu. Công thức này là nền tảng minh họa, không thay thế toán tử của ST-GCN/CTR-GCN. [GCN gốc][r01]

### 9.3. GCN không tự động trả một vector cho cả frame

Một spatial encoder có thể tạo:

$$
X_t\in\mathbb{R}^{J\times C}
\longrightarrow
F_t\in\mathbb{R}^{J\times d}.
$$

Để có vector \(f_t\in\mathbb{R}^{d}\), cần thêm phép gộp, ví dụ:

$$
f_t=\frac{1}{J}\sum_{j=1}^{J}F_{t,j}.
$$

Mean pooling chỉ là ví dụ; có thể dùng attention pooling hoặc gộp theo bộ phận. Khi đã có \(f_1,\ldots,f_T\):

$$
(f_1,\ldots,f_T)
\xrightarrow{\text{Temporal Mamba}}
(h_1,\ldots,h_T).
$$

- **Phân loại clip:** gộp theo thời gian rồi phân loại.
- **Segmentation:** dự đoán từ từng \(h_t\), giữ hoặc khôi phục độ phân giải thời gian.
- **Online:** spatial processing, temporal module, normalization và hậu xử lý đều phải tuân thủ điều kiện không dùng tương lai.

Nếu gộp quá sớm, thông tin ngón tay hoặc tương tác giữa hai bàn tay có thể bị mất. Đây là giả thuyết cần ablation, không phải lý do để cấm pooling.

### 9.4. Tại sao Mamba đáng thử, nhưng chưa chắc thắng?

Khi giữ chiều đặc trưng và kích thước state cố định, selective scan của Mamba tăng chi phí tuyến tính theo số token. Full self-attention chuẩn có phần tính tương tác cặp token tăng bậc hai. [Mamba gốc][r02]

Tuy nhiên, phải xác định **token là gì**:

| Cách biểu diễn | Số token ở temporal/sequence model |
|---|---|
| Một vector cho mỗi frame | \(L=T\) |
| Một token cho mỗi khớp mỗi frame | \(L=TJ\) |
| Nhóm frame/khớp thành token | Phụ thuộc cách nhóm và stride |
| Temporal scan riêng cho từng khớp | \(J\) chuỗi độ dài \(T\) |

So sánh công bằng cần cùng đầu vào, độ dài, cách sampling và ngân sách mô hình. Transformer dạng partition như SkateFormer đã giảm chi phí so với full attention. Với chuỗi ngắn, overhead triển khai và spatial encoder có thể chi phối thời gian chạy. [SkateFormer][p12]

**Đặc biệt:** O(L) không tự chứng minh mô hình nhớ hữu ích mọi thông tin xa; phải đo hiệu quả khi tăng context và khi loại bỏ context xa.

<a id="10-ipn-hand"></a>

## 10. Liên hệ IPN Hand và thiết kế thực nghiệm

### 10.1. Chuyển từ RGB sang skeleton là thay đổi biểu diễn

IPN Hand cung cấp RGB video cho cả đánh giá isolated và continuous gesture recognition. Paper cũng khảo sát các biểu diễn suy ra từ RGB. [IPN Hand][r03]

Một hệ chỉ nhận RGB ở cảm biến nhưng trích skeleton vẫn là:

**RGB-input system → pose estimation → skeleton-based recognition.**

Nó khác mô hình dùng skeleton gốc từ cảm biến độ sâu. Khi báo cáo cần nêu:

1. Pose estimator, phiên bản và trọng số.
2. Keypoint 2D hay 3D ước lượng; định nghĩa confidence.
3. Tần số frame gốc, frame được trích và frame thực sự vào model.
4. Tỉ lệ mất tay/mất khớp, quy tắc điền dữ liệu.
5. Cách giữ ID người/bàn tay qua thời gian.
6. Chi phí toàn hệ thống, bao gồm pose estimation.

MediaPipe Hands là một tài liệu tham khảo cho bước hand tracking; không xem đầu ra 3D ước lượng từ đơn mắt là ground-truth metric 3D. [MediaPipe Hands][r04]

### 10.2. Chuẩn hóa có thể xóa mất thông tin “vẫy tay”

Giả sử \(p_{t,j}\) là tọa độ khớp \(j\), \(p_{t,w}\) là cổ tay và \(s_t>0\) là thang kích thước bàn tay:

$$
q_{t,j}=\frac{p_{t,j}-p_{t,w}}{s_t}.
$$

Biểu diễn \(q_{t,j}\) làm nổi bật hình dạng tương đối của bàn tay. Nhưng nếu cả bàn tay tịnh tiến sang trái/phải mà hình dạng không đổi, phép trừ cổ tay loại bỏ chuyển động tịnh tiến đó.

**Đề xuất thực nghiệm:** giữ cả tọa độ tương đối và một nhánh chuyển động toàn cục:

$$
v^{w}_t=\frac{p_{t,w}-p_{t-1,w}}{\Delta t}.
$$

Ở đây \(\Delta t\) là khoảng thời gian giữa hai mẫu. Nếu bỏ mẫu hoặc thay FPS thì phải cập nhật \(\Delta t\). Đây là đề xuất xử lý dữ liệu, không phải một tính mới đã được chứng minh.

Với robot nhìn người từ xa, hand-only keypoints cũng có thể chưa đủ ổn định. Có thể thử thêm cổ tay, khuỷu và vai, nhưng phải dùng pose estimator hỗ trợ đúng bộ keypoint; mô hình body pose phổ thông không mặc nhiên trả đầy đủ khớp ngón tay.

### 10.3. Câu hỏi nghiên cứu sau khi cập nhật GALFu-Mamba

**Không nên dùng làm tuyên bố tính mới độc lập:**

- “Lần đầu dùng Mamba cho skeleton action recognition.”
- “Lần đầu kết hợp GCN và Mamba.”
- “Lần đầu dùng Mamba cho skeleton temporal segmentation.”

Các hướng có thể khảo sát tiếp, với điều kiện kiểm tra related work hẹp hơn:

| Câu hỏi nghiên cứu | Thí nghiệm phải có |
|---|---|
| SSM causal giữ biên cử chỉ tốt đến đâu khi chỉ có skeleton ước lượng từ RGB? | Causal baseline; đo độ trễ đầu/cuối; kiểm tra không dùng tương lai |
| Context dài có giúp phân biệt gesture với chuyển động nền không? | Tăng context; xáo trộn/cắt context xa; chia nhóm theo độ dài |
| Mô hình chịu mất khớp và jitter như thế nào? | Đánh giá lỗi pose tự nhiên và nhiễu có kiểm soát |
| Nhánh global wrist motion có bổ sung cho local hand shape không? | Ablation local-only/global-only/fusion |
| Hệ có chạy được trên thiết bị mục tiêu với độ trễ ổn định không? | Đo end-to-end, batch 1, p50/p95 latency và memory |

Đây là **các giả thuyết nghiên cứu**, không phải khẳng định khoảng trống đã được xác nhận. GALFu-Mamba cần được đọc toàn văn trước khi chốt đóng góp liên quan local/global fusion.

### 10.4. Bộ baseline tối thiểu

Đề xuất giữ nguyên dữ liệu, spatial encoder và training protocol khi nghiên cứu riêng temporal model:

| Mã thử nghiệm | Mô hình | Mục đích |
|---|---|---|
| B0 | Spatial encoder + temporal pooling | Kiểm tra có cần động lực học chi tiết không |
| B1 | Spatial encoder + TCN | Baseline convolution |
| B2 | Spatial encoder + GRU/LSTM | Baseline recurrent |
| B3 | Spatial encoder + Transformer phù hợp ngân sách | Baseline attention |
| B4 | Spatial encoder + Mamba | Đo đóng góp temporal SSM |
| B5 | B4 + boundary head/loss | Đo đóng góp supervision biên |

Với segmentation, thêm **MS-GCN** và **ME-ST** khi tái lập được protocol. GALFu-Mamba là related work trực tiếp cần bổ sung baseline nếu có thể tiếp cận toàn văn/code. Không gán kết quả tự tái hiện một mô hình gần giống thành kết quả chính thức của paper.

Nếu giữ VideoMamba RGB làm đối chứng, cần ghi rõ nó sử dụng biểu diễn khác. Có thể so chất lượng và chi phí toàn hệ, nhưng không quy mọi chênh lệch cho temporal backbone.

### 10.5. Metric phải đi cùng loại bài toán

| Bài toán/thuộc tính | Chỉ số đề xuất | Cần ghi rõ |
|---|---|---|
| Clip classification | Top-1, macro-F1, confusion matrix | Số lớp, split và sampling |
| Frame prediction | Frame accuracy và macro-F1 | Có/không tính background |
| Chất lượng đoạn | Segmental F1 tại các ngưỡng IoU, edit score | Quy tắc matching và xử lý background |
| Temporal detection | mAP tại temporal IoU | Protocol detection cụ thể |
| Streaming | Delay phát hiện đầu/cuối, false alarms/phút, miss rate | Đơn vị thời gian, tiêu chuẩn khớp sự kiện |
| Hiệu năng | Params, FLOPs, peak memory, latency p50/p95 | Phần cứng, precision, batch, độ dài input |
| Toàn pipeline | FPS và latency có pose estimator | Decode, pose, model, hậu xử lý được tính thế nào |

Không áp dụng segmental edit đơn nhãn một cách máy móc cho dữ liệu có hành động chồng lấn/multi-label. Không kết luận segmentation tốt chỉ từ frame accuracy cao khi background chiếm phần lớn.

### 10.6. Kế hoạch triển khai đề xuất

1. **Chốt bài toán và split.** Xác định clip classification hay continuous segmentation; online hay offline. Chia theo video/người trước khi tạo cửa sổ.
2. **Kiểm toán pose.** Trực quan hóa một tập đại diện, thống kê mất tay, jitter, nhầm ID và che khuất.
3. **Tạo bản dữ liệu tái lập được.** Lưu timestamps, keypoints, confidence/mask, nhãn và cấu hình trích xuất.
4. **Chạy B0–B2.** Nếu mô hình đơn giản đã rất mạnh, xem lại sự cần thiết của context dài trước khi tăng độ phức tạp.
5. **Chạy B3–B4 với ngân sách tương đương.** Báo số frame thực sự dùng; không chỉ độ dài video gốc.
6. **Ablation biên và độ tin cậy.** Chỉ thêm module khi giải quyết được lỗi đã quan sát.
7. **Đánh giá toàn hệ.** Video chưa cắt, batch 1, độ trễ và false alarms.
8. **Viết related work theo câu hỏi.** Nêu rõ so với MS-GCN, ME-ST, GALFu-Mamba và nhóm classification Mamba.

**Sản phẩm tối thiểu:** bảng dữ liệu/protocol; bảng baseline; ablation; đồ thị chất lượng theo context; phân bố latency; ví dụ lỗi và dự đoán theo thời gian.

<a id="11-doc-bai"></a>

## 11. Thứ tự đọc và sản phẩm cần rút ra

Đây là thứ tự đọc đề xuất phục vụ việc hiểu và triển khai; không phải bảng xếp hạng chất lượng paper.

| Bước | Tài liệu | Điều cần ghi ra sau khi đọc |
|---|---|---|
| 1 | H-RNN, ST-LSTM, VA-LSTM | Cách biểu diễn skeleton; vấn đề view/noise; causal hay hai chiều |
| 2 | ST-GCN | Tensor input, adjacency, spatial block và temporal block |
| 3 | 2s-AGCN, CTR-GCN | Topology học thế nào; khác joint/bone stream |
| 4 | STA-Res-TCN, DSTA-Net | Những điểm riêng của hand gesture và attention |
| 5 | STTFormer, SkateFormer | Tokenization, partition, độ phức tạp thực tế |
| 6 | Mamba gốc, Simba, TSkel-Mamba | Trục scan; state; vị trí của Mamba; ablation |
| 7 | SkelMamba, GCN-Mamba | Nhiều stream; trade-off tham số và độ chính xác |
| 8 | MS-GCN, LAC, ME-ST | Đầu ra dense, supervision, refinement và metric |
| 9 | GALFu-Mamba | Đối chiếu trực tiếp trước khi tuyên bố novelty về segmentation |
| 10 | IPN Hand, MediaPipe Hands, MS-Temba | Dữ liệu liên tục, lỗi pose và cách xử lý video dài |

**Nhánh đọc ngay có PDF mở:** H-RNN → ST-GCN → 2s-AGCN → CTR-GCN → DSTA-Net → SkateFormer → Simba → TSkel-Mamba → MS-GCN → LAC.

**Nhánh cần tìm thêm toàn văn:** SkeMamba, SkeletonMamba, ME-GCN, motion-aware ST-GCN, ME-ST, GALFu-Mamba. Với ME-ST và SkeletonMamba, repository tác giả vẫn giúp kiểm tra cấu hình triển khai dù toàn văn chưa mở.

### Mẫu ghi chú cho mỗi paper

| Trường | Câu hỏi cần trả lời |
|---|---|
| Nguồn | Tên, tác giả, venue, năm, DOI, URL phiên bản đọc |
| Research question | Paper giải quyết hạn chế cụ thể nào? |
| Input | Skeleton gốc/ước lượng; 2D/3D/confidence; số khớp/người |
| Spatial module | Graph cố định, adaptive graph, attention hay vector? |
| Temporal module | TCN, RNN, Transformer, Mamba; đơn/hai chiều? |
| Sequence | Số frame vào mạng; stride; có downsample hoặc window không? |
| Output | Nhãn clip, frame, multi-label hay interval? |
| Training | Pretraining, augmentation, số luồng, loss |
| Evaluation | Dataset/split, metric, mean/std, điều kiện phần cứng |
| Ablation | Có tách đóng góp từng module không? |
| Limitation | Điều kiện nào paper chưa kiểm tra? |
| Reproduction | Code, weights, environment và khả năng chạy lại |

<a id="12-bo-sung"></a>

## 12. Tài liệu bổ sung và giới hạn khảo sát

### R01. Nền tảng GCN

**Thomas N. Kipf, Max Welling.** *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017; preprint 2016.  
[Trang bài][r01] · [PDF mở](https://arxiv.org/pdf/1609.02907).

Dùng để hiểu adjacency, self-loop, degree normalization và phép cập nhật đặc trưng node. Đây là tài liệu GCN tổng quát, không phải paper skeleton recognition.

### R02. Nền tảng Mamba

**Albert Gu, Tri Dao.** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. Preprint 2023, bản cập nhật 2024.  
[Trang bài][r02] · [PDF mở](https://arxiv.org/pdf/2312.00752).

Dùng để hiểu selective SSM, quan hệ giữa input và tham số chọn lọc, cùng cách scan. Không chuyển các số tăng tốc trong ngôn ngữ thành tuyên bố tăng tốc cho skeleton nếu chưa đo.

### R03. IPN Hand

**Gibran Benitez-Garcia, Jesus Olivares-Mercado, Gabriel Sanchez-Perez, Keiji Yanai.** *IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition*. ICPR 2020.  
[Trang bài][r03] · [PDF mở](https://arxiv.org/pdf/2005.02134) · [Code/dữ liệu](https://github.com/GibranBenitez/IPN-hand).

Dùng để xác nhận định nghĩa isolated/continuous evaluation, các đoạn non-gesture và cấu trúc dữ liệu. Nếu chỉ dùng RGB đầu vào, ghi rõ các biểu diễn suy ra được sử dụng.

### R04. MediaPipe Hands

**Fan Zhang và cộng sự.** *MediaPipe Hands: On-device Real-time Hand Tracking*. Preprint 2020.  
[Trang bài][r04] · [PDF mở](https://arxiv.org/pdf/2006.10214).

Dùng để hiểu bộ trích keypoint trước recognizer. Việc dùng pose estimator có sẵn không loại bỏ nhu cầu đánh giá lỗi pose trên dữ liệu thực tế.

### 12.1. Những điều chưa được xác minh hoàn toàn

- Chưa đọc được toàn văn sáu mục được đánh dấu thiếu PDF mở.
- Chưa tái lập kết quả của các mô hình trong khảo sát.
- Chưa xây dựng bảng accuracy chung do khác modality, số luồng, split, pretraining, frame sampling và phiên bản.
- Chưa kết luận research gap hẹp về causal hand segmentation từ RGB; cần tiếp tục đối chiếu GALFu-Mamba và literature về online gesture spotting.
- Chưa xác minh mọi preprint có hay không có một phiên bản phản biện mới hơn dưới tiêu đề khác.
- Không xem absence trong kết quả tìm kiếm là chứng minh absence của nghiên cứu.

### 12.2. Các chỉnh sửa so với nội dung trao đổi trước

| Nội dung cần chỉnh | Cách hiểu dùng trong tài liệu |
|---|---|
| Chưa thấy Mamba cho skeleton temporal segmentation | Đã tìm thấy GALFu-Mamba, IEEE SWC 2025 |
| Chuỗi dài “thực sự” phải là segmentation | Độ dài chuỗi và kiểu đầu ra là hai trục độc lập |
| Các giai đoạn thay thế nhau hoàn toàn | RNN, TCN, graph, attention và SSM tiếp tục tồn tại/kết hợp |
| GCN luôn chỉ spatial | Nhiều kiến trúc graph-based có temporal module hoặc graph không–thời gian |
| Qua GCN là tự có vector 256-D mỗi frame | Cần pooling/projection nếu muốn gộp chiều joint |
| Mamba luôn tốt hơn Transformer vì O(L) | Cần kiểm chứng theo tokenization, context và implementation |
| Độ sâu ước lượng từ RGB tương đương depth sensor | Phải phân biệt nguồn đo, hệ tọa độ, đơn vị và sai số |
| Toàn bộ cải thiện Simba là do Mamba | Cần đọc ablation của U-ShiftGCN |
| SkeletonMamba chỉ cần ghi một năm 2026 | Hội nghị/online 2026 nhưng metadata trích dẫn Springer hiện dùng 2027 |

**Quy tắc sử dụng tài liệu:** dùng các liên kết bên dưới/ở từng mục để đọc bản gốc; khi viết báo cáo hoặc bài báo, trích dẫn paper gốc và đúng phiên bản, không trích tài liệu tổng hợp này thay cho bằng chứng thực nghiệm.

---

## Danh mục liên kết

Các định nghĩa liên kết dưới đây là một phần của cú pháp Markdown; chúng giúp giữ các bảng gọn và vẫn mở được bài báo.

[p01]: https://openaccess.thecvf.com/content_cvpr_2015/html/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.html
[p01-pdf]: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.pdf
[p02]: https://arxiv.org/abs/1607.07043
[p02-pdf]: https://arxiv.org/pdf/1607.07043
[p03]: https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html
[p03-pdf]: https://arxiv.org/pdf/1703.08274
[p04]: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Kim_Interpretable_3D_Human_CVPR_2017_paper.html
[p04-pdf]: https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/papers/Kim_Interpretable_3D_Human_CVPR_2017_paper.pdf
[p05]: https://aaai.org/papers/12328-spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/
[p05-pdf]: https://arxiv.org/pdf/1801.07455
[p06]: https://doi.org/10.1007/978-3-030-11024-6_18
[p06-pdf]: https://image.ee.tsinghua.edu.cn/pdf/2018_hjx_ECCVW.pdf
[p07]: https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
[p07-pdf]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf
[p08]: https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.html
[p08-pdf]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf
[p09]: https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html
[p09-pdf]: https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.pdf
[p10]: https://openaccess.thecvf.com/content/ACCV2020/html/Shi_Decoupled_Spatial-Temporal_Attention_Network_for_Skeleton-Based_Action-Gesture_Recognition_ACCV_2020_paper.html
[p10-pdf]: https://openaccess.thecvf.com/content/ACCV2020/papers/Shi_Decoupled_Spatial-Temporal_Attention_Network_for_Skeleton-Based_Action-Gesture_Recognition_ACCV_2020_paper.pdf
[p11]: https://arxiv.org/abs/2201.02849
[p11-pdf]: https://arxiv.org/pdf/2201.02849
[p12]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5796_ECCV_2024_paper.php
[p12-pdf]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05796.pdf
[p13]: https://arxiv.org/abs/2404.07645
[p13-pdf]: https://arxiv.org/pdf/2404.07645
[p14]: https://arxiv.org/abs/2411.19544
[p14-pdf]: https://arxiv.org/pdf/2411.19544
[p15]: https://doi.org/10.1007/s00371-025-03950-5
[p16]: https://www.mdpi.com/2079-9292/14/18/3610
[p16-pdf]: https://www.mdpi.com/2079-9292/14/18/3610/pdf
[p17]: https://arxiv.org/abs/2512.11503
[p17-pdf]: https://arxiv.org/pdf/2512.11503
[p18]: https://www.jait.us/show-272-1908-1.html
[p18-pdf]: https://www.jait.us/articles/2026/JAIT-V17N7-1269.pdf
[p19]: https://doi.org/10.1007/978-3-032-31404-8_8
[p20]: https://onlinelibrary.wiley.com/doi/10.1002/cav.70164
[p21]: https://arxiv.org/abs/2202.01727
[p21-pdf]: https://arxiv.org/pdf/2202.01727
[p22]: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_LAC_-_Latent_Action_Composition_for_Skeleton-based_Action_Segmentation_ICCV_2023_paper.html
[p22-pdf]: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_LAC_-_Latent_Action_Composition_for_Skeleton-based_Action_Segmentation_ICCV_2023_paper.pdf
[p23]: https://doi.org/10.1016/j.neucom.2024.127482
[p24]: https://pubmed.ncbi.nlm.nih.gov/40327485/
[p25]: https://ieeexplore.ieee.org/abstract/document/11394985/
[p26]: https://openaccess.thecvf.com/content/CVPR2026/html/Sinha_MS-Temba_Multi-Scale_Temporal_Mamba_for_Understanding_Long_Untrimmed_Videos_CVPR_2026_paper.html
[p26-pdf]: https://arxiv.org/pdf/2501.06138
[r01]: https://arxiv.org/abs/1609.02907
[r02]: https://arxiv.org/abs/2312.00752
[r03]: https://arxiv.org/abs/2005.02134
[r04]: https://arxiv.org/abs/2006.10214

