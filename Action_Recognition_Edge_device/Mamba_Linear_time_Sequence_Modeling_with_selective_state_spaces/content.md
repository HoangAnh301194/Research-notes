# Nội dung slide báo cáo: Mamba – Mô hình hóa chuỗi thời gian tuyến tính với Không gian Trạng thái Chọn lọc

> **Nguồn tham khảo gốc:** Albert Gu, Tri Dao. *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*. arXiv:2312.00752, 2023.
> **Mã nguồn công bố:** https://github.com/state-spaces/mamba
> **File dịch và chú giải:** `Paper_translating.md` (cùng thư mục)

---

## SLIDE 01 – Trang bìa

**Tiêu đề chính:**
Mamba: Mô hình hóa chuỗi thời gian tuyến tính với Không gian Trạng thái Chọn lọc

**Tiêu đề phụ:**
Tổng quan và phân tích bài báo: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*

**Thông tin:**
- Tác giả bài báo: Albert Gu (CMU), Tri Dao (Princeton)
- Năm công bố: 2023 (arXiv 2312.00752)
- Người trình bày: [Họ tên] – [Khoa / Nhóm NCKH]
- Hướng nghiên cứu: Nhận dạng hành động / Triển khai mô hình trên thiết bị biên

---

## SLIDE 02 – Nội dung trình bày

**Cấu trúc báo cáo:**

| # | Phần | Nội dung |
|---|---|---|
| 1 | Bối cảnh và động lực | Hạn chế của Transformer; vai trò của SSM |
| 2 | Nền tảng lý thuyết | Mô hình không gian trạng thái có cấu trúc (S4) |
| 3 | Đóng góp kỹ thuật | Cơ chế chọn lọc; thuật toán phần cứng; kiến trúc |
| 4 | Đánh giá thực nghiệm | Ngôn ngữ, DNA, âm thanh, tổng hợp |
| 5 | Phân tích hiệu quả | Tốc độ và bộ nhớ |
| 6 | Thảo luận và kết luận | Hạn chế; định hướng ứng dụng |

---

## SLIDE 03 – Bối cảnh và động lực nghiên cứu

**Vấn đề nghiên cứu:**
Các mô hình nền tảng (*foundation models*) hiện đại chủ yếu sử dụng kiến trúc Transformer với cơ chế *self-attention* làm thành phần cốt lõi. Mặc dù cơ chế này cho phép định tuyến thông tin dày đặc trong cửa sổ ngữ cảnh và đạt hiệu năng vượt trội trên nhiều tác vụ, song nó mang theo hai hạn chế cơ bản:

1. **Độ phức tạp tính toán bậc hai:** Chi phí huấn luyện tăng theo O(L²) so với độ dài chuỗi L, gây ra tắc nghẽn nghiêm trọng khi xử lý chuỗi dài.
2. **Bộ nhớ suy luận tuyến tính:** Quá trình suy luận tự hồi quy (*autoregressive inference*) yêu cầu lưu trữ toàn bộ *KV cache*, dẫn đến mức tiêu thụ bộ nhớ tỷ lệ tuyến tính với độ dài ngữ cảnh.

**Lớp mô hình thay thế:**
Các mô hình không gian trạng thái có cấu trúc (*Structured State Space Models* – SSMs) đã được phát triển như một giải pháp thay thế với độ phức tạp tuyến tính. Tuy nhiên, các SSM trước đây vẫn chưa đạt chất lượng tương đương Transformer trên các miền dữ liệu rời rạc (*discrete modalities*) như ngôn ngữ.

**Mục tiêu của bài báo:**
Đề xuất lớp mô hình Selective SSM vừa đạt chất lượng Transformer vừa duy trì độ phức tạp tính toán tuyến tính theo độ dài chuỗi.

---

## SLIDE 04 – Nền tảng lý thuyết: Mô hình không gian trạng thái có cấu trúc

**Định nghĩa:**
Các mô hình S4 được xây dựng trên cơ sở hệ thống liên tục (*continuous-time system*) ánh xạ tín hiệu đầu vào x(t) ∈ ℝ sang đầu ra y(t) ∈ ℝ thông qua trạng thái ẩn h(t) ∈ ℝᴺ:

```
h'(t) = A·h(t) + B·x(t)          (phương trình trạng thái liên tục)
y(t)  = C·h(t)                    (phương trình đầu ra)
```

**Rời rạc hóa (Discretization):**
Tham số liên tục (Δ, A, B) được chuyển đổi sang dạng rời rạc (Ā, B̄) thông qua quy tắc *zero-order hold* (ZOH):
```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹(exp(ΔA) − I)·ΔB
```

**Hai chế độ tính toán tương đương:**

| Chế độ | Công thức | Ứng dụng | Độ phức tạp |
|---|---|---|---|
| Hồi quy (*Recurrent*) | hₜ = Ā·hₜ₋₁ + B̄·xₜ | Suy luận tự hồi quy | O(1)/bước |
| Tích chập (*Convolution*) | y = x * K̄ | Huấn luyện song song | O(L log L) |

**Tính bất biến theo thời gian (LTI):**
Toàn bộ các SSM trước đây đều tuân theo tính chất *Linear Time Invariance* (LTI), tức các tham số (Δ, A, B, C) là hằng số theo thời gian, không phụ thuộc vào nội dung đầu vào.

---

## SLIDE 05 – Giới hạn của SSM bất biến theo thời gian (LTI)

**Hạn chế cốt lõi:**
Tính LTI khiến các SSM truyền thống không có khả năng thực hiện *content-based reasoning* – suy luận dựa trên nội dung của chuỗi đầu vào.

**Phân tích theo hai góc nhìn:**

- **Góc nhìn hồi quy:** Phép chuyển trạng thái (Ā, B̄) cố định không cho phép mô hình lựa chọn thông tin quan trọng hay loại bỏ nhiễu tùy theo token hiện tại.
- **Góc nhìn tích chập:** Kernel tích chập toàn cục cố định (*static global convolution kernel*) không phân biệt được nội dung, chỉ phân biệt vị trí tương đối.

**Hệ quả thực tế:**
- Mô hình LTI có thể giải bài toán *Copying* thông thường (khoảng cách cố định) nhờ kernel độ dài phù hợp.
- Nhưng thất bại với *Selective Copying* (khoảng cách ngẫu nhiên) vì không thể phân biệt token quan trọng và nhiễu theo nội dung.

**Đánh đổi cơ bản trong mô hình hóa chuỗi:**
Hiệu quả tính toán và chất lượng dự đoán của mô hình chuỗi phụ thuộc vào khả năng nén ngữ cảnh vào trạng thái ẩn. Mô hình hiệu quả cần trạng thái nhỏ (suy luận nhanh), trong khi mô hình hiệu quả dự đoán cần trạng thái lớn đủ chứa thông tin ngữ cảnh cần thiết.

---

## SLIDE 06 – Động lực: Hai bài toán tổng hợp

**Mục đích:**
Hai bài toán tổng hợp được sử dụng để làm rõ điểm yếu của LTI SSM và xác định yêu cầu thiết kế cho cơ chế chọn lọc.

**Bài toán 1 – Selective Copying:**
- **Mô tả:** Cho chuỗi đầu vào gồm các token dữ liệu xen kẽ với token nhiễu tại các vị trí ngẫu nhiên. Mô hình được yêu cầu sao chép lại chỉ các token dữ liệu theo đúng thứ tự.
- **Yêu cầu:** Mô hình phải thực hiện *content-aware reasoning* – nhận biết token nào cần ghi nhớ dựa trên nội dung, không phải vị trí cố định.
- **Kết quả với LTI:** Thất bại do kernel tích chập cố định không thể thích ứng với khoảng cách thay đổi.

**Bài toán 2 – Induction Heads:**
- **Mô tả:** Nếu mô hình đã quan sát cặp bigram [A, B] trong chuỗi, khi gặp lại token A, mô hình phải dự đoán B tiếp theo.
- **Yêu cầu:** Truy hồi thông tin kết hợp (*associative recall*) – kỹ năng then chốt của mô hình ngôn ngữ lớn cho phép học theo ngữ cảnh (*in-context learning*).
- **Ý nghĩa nghiên cứu:** Mô hình phải ngoại suy tốt sang chuỗi dài hơn nhiều so với khi huấn luyện.

**Kết luận:** Tính chọn lọc (*selectivity*) được đề xuất như nguyên lý nền tảng: khả năng lựa chọn thông tin theo ngữ cảnh khi đưa đầu vào vào trạng thái tuần tự.

---

## SLIDE 07 – Đóng góp thứ nhất: Cơ chế Chọn lọc

**Ý tưởng cốt lõi:**
Cho phép các tham số SSM là hàm của đầu vào (*functions of the input*) thay vì hằng số. Thay đổi đơn giản nhưng có tác động căn bản đến khả năng biểu diễn của mô hình.

**Tham số hóa chọn lọc:**

| Tham số | SSM truyền thống (S4) | Selective SSM (S6) |
|---|---|---|
| B | Hằng số: (D, N) | Phụ thuộc đầu vào: (B, L, N) |
| C | Hằng số: (D, N) | Phụ thuộc đầu vào: (B, L, N) |
| Δ | Hằng số: (D) | Phụ thuộc đầu vào: (B, L, D) |
| A | Hằng số: (D, N) | Hằng số (giữ nguyên) |

Cụ thể, các hàm chọn lọc được định nghĩa:
```
s_B(x) = Linear_N(x)
s_C(x) = Linear_N(x)
s_Δ(x) = Broadcast_D(Linear_1(x)),  τ_Δ = softplus
```

**Tác động kỹ thuật:**
Sự phụ thuộc vào đầu vào làm cho tham số có thêm chiều độ dài L, khiến mô hình chuyển từ *time-invariant* sang *time-varying*. Điều này phá vỡ tính tương đương với tích chập (phương trình 3), đặt ra thách thức về hiệu quả tính toán được giải quyết ở đóng góp tiếp theo.

---

## SLIDE 08 – So sánh Thuật toán S4 và S6

**Thuật toán S4 – SSM bất biến theo thời gian:**

```
Đầu vào : x ∈ ℝ^{B×L×D}
Tham số : A ∈ ℝ^{D×N},  B ∈ ℝ^{D×N},  C ∈ ℝ^{D×N},  Δ ∈ ℝ^D
Bước 1  : Ā, B̄ = discretize(Δ, A, B)          ← tham số cố định
Bước 2  : y = SSM(Ā, B̄, C)(x)                  ← tính bằng convolution
```

**Thuật toán S6 – Selective SSM (biến thiên theo thời gian):**

```
Đầu vào : x ∈ ℝ^{B×L×D}
Tham số : A ∈ ℝ^{D×N}                           ← cố định
          B ← s_B(x) ∈ ℝ^{B×L×N}               ← phụ thuộc đầu vào
          C ← s_C(x) ∈ ℝ^{B×L×N}               ← phụ thuộc đầu vào
          Δ ← τ_Δ(Param + s_Δ(x)) ∈ ℝ^{B×L×D} ← phụ thuộc đầu vào
Bước 1  : Ā, B̄ = discretize(Δ, A, B)          ← biến đổi theo thời gian
Bước 2  : y = SSM(Ā, B̄, C)(x)                  ← chỉ tính bằng recurrence + scan
```

**Kết quả đánh giá trên bài toán Selective Copying:**

| Mô hình | Kiến trúc | Lớp SSM | Độ chính xác |
|---|---|---|---|
| S4 | Không gate | S4 | 18,3% |
| H3 | H3 | S4 | 57,0% |
| Hyena | H3 | Hyena | 30,1% |
| **Mamba** | **Mamba** | **S6** | **99,8%** |

Kết quả xác nhận rằng cơ chế chọn lọc – chứ không phải kiến trúc có gate – là yếu tố quyết định hiệu năng.

---

## SLIDE 09 – Đóng góp thứ hai: Thuật toán thân thiện với phần cứng

**Thách thức kỹ thuật:**
Selective SSM không còn tương đương với tích chập, do đó không thể sử dụng các triển khai convolution hiệu quả. Nếu tính toán thô sơ, trạng thái mở rộng h có shape (B, L, D, N) sẽ tiêu tốn lượng bộ nhớ không chấp nhận được trên GPU.

**Giải pháp: Ba kỹ thuật kết hợp**

**① Kernel Fusion (hợp nhất kernel):**
Thay vì lưu tất cả trạng thái trung gian vào HBM (High Bandwidth Memory – bộ nhớ chậm), thuật toán tải tham số (Δ, A, B, C) từ HBM vào SRAM (bộ nhớ nhanh trên chip), thực hiện toàn bộ discretization và recurrence trong SRAM, rồi chỉ ghi kết quả đầu ra (B, L, D) trở lại HBM. Điều này giảm lượng I/O bộ nhớ theo hệ số O(N).

**② Parallel Scan (quét song song):**
Mặc dù recurrence có tính tuần tự về mặt toán học, thuật toán *parallel associative scan* cho phép song song hóa hiệu quả, đảm bảo độ phức tạp tuyến tính O(L) theo độ dài chuỗi.

**③ Recomputation (tính lại):**
Thay vì lưu các trạng thái trung gian phục vụ lan truyền ngược, thuật toán tính lại chúng từ đầu vào trong quá trình backward pass, tránh chi phí bộ nhớ O(BLND).

**Kết quả đo lường:**
- Nhanh hơn triển khai scan chuẩn PyTorch: **20–40 lần**
- Nhanh hơn FlashAttention-2 khi độ dài chuỗi vượt **2.048 token**
- Yêu cầu bộ nhớ tương đương Transformer tối ưu bằng FlashAttention

---

## SLIDE 10 – Đóng góp thứ ba: Kiến trúc Mamba

**Thiết kế tổng thể:**
Kiến trúc Mamba tích hợp Selective SSM vào một khối (*block*) mạng neural đơn giản và đồng nhất, không sử dụng attention hay các khối MLP riêng biệt.

**Cấu trúc Mamba Block:**
```
Đầu vào x
    ├── Nhánh chính: Linear → Conv1D → SSM (S6) → × [nhân theo từng phần tử]
    └── Nhánh gate:  Linear → SiLU activation
         ↓
    Đầu ra: Linear projection + Residual + LayerNorm
```

**Nguyên lý thiết kế:**
- Hợp nhất H3 block (nền tảng của phần lớn kiến trúc SSM) và MLP block (phổ biến trong Transformer) thành một khối duy nhất, lấy cảm hứng từ *Gated Attention Unit*.
- Expansion factor E = 2 cho từng khối; xếp chồng đồng nhất thay vì xen kẽ hai loại khối khác nhau.
- Tổng tham số: 12D² mỗi cặp khối, tương đương cặp MHA + MLP trong Transformer.

**So sánh kiến trúc:**

| Đặc điểm | Transformer | H3 | **Mamba** |
|---|---|---|---|
| Cơ chế Attention | Có | Không | **Không** |
| Cấu trúc khối | MHA + MLP xen kẽ | H3 + MLP xen kẽ | **Mamba block đồng nhất** |
| SSM bên trong | – | S4 (LTI) | **S6 (Selective)** |
| Activation | GELU/SwiGLU | – | **SiLU/Swish** |

---

## SLIDE 11 – Phân tích tính chất của cơ chế Chọn lọc

**Ba cơ chế hành vi nổi bật:**

**① Lọc không gian biến đổi (Variable Spacing):**
Mô hình có khả năng bỏ qua các token nhiễu hoặc không liên quan bằng cách cho Δₜ nhỏ, tương đương gate gₜ → 0 trong dạng RNN hóa.

**② Lọc ngữ cảnh (Context Filtering):**
Không giống LTI SSM – vốn không thể bỏ qua ngữ cảnh không liên quan và thường không cải thiện khi thêm context – Selective SSM có thể *reset* trạng thái bất kỳ lúc nào, khiến hiệu năng cải thiện đơn điệu khi ngữ cảnh dài hơn.

**③ Đặt lại ranh giới (Boundary Resetting):**
Khi nhiều chuỗi độc lập được ghép nối trong quá trình huấn luyện, mô hình tự học cách reset trạng thái tại ranh giới chuỗi (Δₜ → ∞), tương đương với attention mask trong Transformer nhưng theo cách học được.

**Mối liên hệ với cơ chế gate RNN – Định lý 1:**
Khi N=1, A=-1, B=1, s_Δ = Linear(x), τ_Δ = softplus, selective SSM có dạng đúng bằng RNN gate:
```
gₜ = σ(Linear(xₜ))
hₜ = (1 - gₜ)·hₜ₋₁ + gₜ·xₜ
```
Điều này xác lập rằng SSM với cơ chế chọn lọc là sự tổng quát hóa có nguyên lý của LSTM/GRU thông qua rời rạc hóa.

---

## SLIDE 12 – Đánh giá thực nghiệm: Bài toán tổng hợp

**Thiết lập thực nghiệm:**
- Selective Copying: chuỗi dài 4.096, vocabulary size 16, 16 token cần ghi nhớ, mô hình 2 layer, D = 64.
- Induction Heads: mô hình 2 layer, huấn luyện ở sequence length 256, đánh giá từ 2⁶ = 64 đến 2²⁰ = 1.048.576.

**Kết quả Selective Copying:**

| Mô hình | Kiến trúc | SSM Layer | Accuracy |
|---|---|---|---|
| S4 | Không gate | S4 | 18,3% |
| H3 | H3 | S4 | 57,0% |
| Hyena | H3 | Hyena | 30,1% |
| – | H3 | S6 | 99,7% |
| **Mamba** | **Mamba** | **S6** | **99,8%** |

**Kết quả Induction Heads – Khả năng ngoại suy:**
- Mamba đạt accuracy hoàn hảo ở mọi độ dài từ 2⁶ đến 2²⁰ (hơn 4.000 lần dài hơn so với lúc huấn luyện).
- Mọi biến thể attention bị giới hạn bộ nhớ ở 2¹⁴ = 16.384, không thể đánh giá vượt ngưỡng này.
- H3 và Hyena (LTI SSM) mất khả năng khái quát khi độ dài vượt 2× so với lúc huấn luyện.

**Nhận xét:** Đây là bằng chứng thực nghiệm trực tiếp cho thấy cơ chế chọn lọc – không phải kiến trúc gate – mới là yếu tố quyết định khả năng mô hình hóa chuỗi dài.

---

## SLIDE 13 – Đánh giá thực nghiệm: Mô hình hóa ngôn ngữ – Scaling Law

**Thiết lập:**
- Dữ liệu huấn luyện: The Pile (300B token, GPT-NeoX tokenizer)
- Phạm vi kích thước mô hình: 125M đến 1,3B tham số
- Giao thức: Chinchilla scaling law (số token huấn luyện tăng tỷ lệ với kích thước mô hình)
- Baseline chính: Transformer (GPT-3), Transformer++ (PaLM/LLaMA recipe), RWKV, RetNet, Hyena

**Kết quả scaling law:**
Mamba là mô hình không sử dụng attention *đầu tiên* đạt hiệu năng ngang bằng Transformer++ – công thức Transformer rất mạnh dựa trên LLaMA. Ưu thế của Mamba tăng rõ khi context length tăng từ 2K lên 8K token.

**Kết quả đánh giá zero-shot downstream:**

| Kích thước | Mô hình | Avg. 6 benchmarks |
|---|---|---|
| ~1,4B | Pythia-1,4B | 55,2% |
| ~1,4B | RWKV-1,5B | 54,3% |
| ~1,4B | **Mamba-1,4B** | **59,7%** |
| ~3B | Pythia-2,8B | 59,1% |
| ~3B | RWKV-3B | 59,6% |
| ~3B | **Mamba-2,8B** | **63,3%** |

*Các benchmark: LAMBADA (PPL, Acc), HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande.*

**Nhận xét quan trọng:** Mamba-3B vượt Pythia-7B (lớn hơn gấp đôi về số tham số) trên tập hợp các benchmark common sense reasoning.

---

## SLIDE 14 – Đánh giá thực nghiệm: Mô hình hóa chuỗi DNA

**Bối cảnh:**
DNA có đặc điểm tương tự ngôn ngữ: chuỗi token rời rạc trên vocabulary hữu hạn (4 nucleotide), đồng thời nổi tiếng đòi hỏi mô hình hóa phụ thuộc xa. Nghiên cứu sử dụng dataset HG38 (genome người, khoảng 4,5B cặp base).

**Thí nghiệm 1 – Scaling theo kích thước mô hình:**
- Context length cố định: 1.024; phạm vi kích thước: ~200K đến ~40M tham số.
- **Kết quả:** Mamba scale tốt hơn cả HyenaDNA và Transformer++; đạt chất lượng tương đương với số tham số ít hơn 3–4 lần.

**Thí nghiệm 2 – Scaling theo độ dài chuỗi:**
- Kích thước mô hình cố định (~1,3M–1,4M tham số); context: 2¹⁰ đến 2²⁰ = 1.048.576.
- **Kết quả Mamba:** Perplexity cải thiện đơn điệu khi context dài hơn tới 1 triệu cặp base.
- **Kết quả HyenaDNA (LTI):** Hiệu năng giảm khi context tăng – xác nhận lý thuyết về hạn chế của mô hình LTI.

**Thí nghiệm 3 – Phân loại loài:**
- Tác vụ: phân loại 5 loài vượn lớn (*great apes*) từ đoạn DNA ngẫu nhiên (chia sẻ ~99% genome).
- **Kết quả:** Mamba vượt HyenaDNA ở mọi độ dài context, đặc biệt ưu thế tăng theo context.

---

## SLIDE 15 – Đánh giá thực nghiệm: Mô hình hóa tín hiệu âm thanh

**Kiến trúc thử nghiệm:**
Sử dụng backbone U-Net của SaShiMi, thay thế các khối S4+MLP bằng Mamba block để đánh giá đóng góp của selective SSM.

**Thí nghiệm 1 – Pretraining autoregressive:**
- Dataset: YouTubeMix (4 giờ audio piano, 16.000 Hz, tối đa 1 phút mỗi clip)
- Metric: Bits per byte (BPB)
- **Kết quả:** Mamba vượt SaShiMi ở mọi context length và tiếp tục cải thiện đến chuỗi gần 1 triệu samples (~1 phút audio).

**Thí nghiệm 2 – Sinh tiếng nói (SC09):**
- Dataset: SC09 (clip 1 giây, 16.000 Hz, các chữ số "zero"–"nine")
- **Kết quả Mamba-6,1M:** FID = 0,94 – vượt SaShiMi (FID = 1,99) và DiffWave+SaShiMi (FID = 1,42) với ít tham số hơn nhiều.
- **Kết quả Mamba-24,3M:** FID = 0,67 – giảm hơn một nửa so với state-of-the-art trước đó.

**Phân tích ablation kiến trúc:**
Mamba block nhất quán tốt hơn S4+MLP tại outer blocks; tại center blocks: Mamba > S4+MLP > MHA+MLP.

---

## SLIDE 16 – Hiệu quả tính toán: Tốc độ và bộ nhớ

**Benchmark tốc độ:**

*Huấn luyện – Selective scan:*
- Nhanh hơn scan chuẩn PyTorch: **20–40 lần**
- Nhanh hơn FlashAttention-2 khi độ dài chuỗi > **2.048 token**
- Độ phức tạp tuyến tính O(L) theo độ dài chuỗi

*Suy luận – Throughput:*
- Mamba đạt throughput cao hơn Transformer cùng kích thước: **4–5 lần**
- Không có KV cache cho phép batch size lớn hơn đáng kể trong inference
- Mamba-6,9B (chưa huấn luyện) có throughput cao hơn Transformer-1,3B

**So sánh tổng hợp về hiệu quả:**

| Tiêu chí | Transformer | Mamba |
|---|---|---|
| Huấn luyện – FLOPs | O(L²) | **O(L)** |
| Suy luận – bộ nhớ | O(L) (KV cache) | **O(1)** |
| Throughput suy luận | 1× | **~5×** |
| Context tối đa (thực tiễn) | ~32K token | **>1M token** |

**Yêu cầu bộ nhớ activation trong huấn luyện:**
Mỗi selective SSM layer lưu ~16 byte activation/token; hai SSM layers tương đương một attention layer cộng một MLP layer trong FlashAttention.

---

## SLIDE 17 – Phân tích ablation: Các thành phần của Mamba

**Ablation 1 – Kiến trúc và lớp SSM:**
- Các SSM LTI (S4 phức, S4 thực): perplexity ~ 10,3–10,5
- Thay bằng S6 (selective): perplexity giảm xuống **8,69** – cải thiện ~2 điểm
- Kiến trúc Mamba và H3 cho kết quả tương đương; Mamba nhỉnh hơn khi dùng S6

**Ablation 2 – Tầm quan trọng của từng tham số selective:**

| Selective Δ | Selective B | Selective C | Perplexity |
|---|---|---|---|
| Không | Không | Không | 10,93 |
| Có | Không | Không | 9,81 |
| Không | Có | Không | 10,15 |
| Không | Không | Có | 9,98 |
| **Có** | **Có** | **Có** | **8,71** |

Δ là tham số selective quan trọng nhất (do liên hệ với RNN gating); kết hợp cả ba tạo hiệu ứng hiệp đồng.

**Ablation 3 – Chiều trạng thái N:**
- B, C hằng số: tăng N từ 1→16, perplexity chỉ giảm từ 9,88→9,81 (không đáng kể)
- B, C selective: tăng N từ 1→16, perplexity giảm mạnh từ **9,73→8,71** (cải thiện 1,0 điểm với chỉ ~1% tham số thêm)

---

## SLIDE 18 – So sánh tổng quan các kiến trúc mô hình chuỗi

**Bảng so sánh toàn diện:**

| Tiêu chí đánh giá | Transformer | LSTM / GRU | S4 (LTI SSM) | **Mamba** |
|---|---|---|---|---|
| Chất lượng ngôn ngữ | Rất tốt | Kém | Trung bình | **Rất tốt** |
| Chất lượng chuỗi liên tục | Tốt | Trung bình | Tốt | **Tốt** |
| Độ phức tạp huấn luyện | O(L²) | O(L) | O(L) | **O(L)** |
| Độ phức tạp suy luận | O(L) | O(1) | O(1) | **O(1)** |
| Suy luận nhận biết nội dung | Có | Hạn chế | Không | **Có** |
| Ngữ cảnh dài (>100K) | Không | Không hiệu quả | Không hiệu quả | **Có (>1M)** |
| Không cần KV cache | Không | Có | Có | **Có** |
| Ngoại suy độ dài chuỗi | Khó | Một phần | Không | **Có** |

**Vị trí của Mamba trong không gian kiến trúc:**
Mamba lần đầu tiên đạt được đồng thời chất lượng Transformer và hiệu quả tuyến tính của RNN/SSM trên các miền dữ liệu rời rạc như ngôn ngữ, đồng thời duy trì ưu thế trên chuỗi liên tục dài như audio và DNA.

---

## SLIDE 19 – Thảo luận: Hạn chế và hướng nghiên cứu tương lai

**Hạn chế được tác giả thừa nhận:**

**① Phạm vi đánh giá:**
Thực nghiệm giới hạn ở quy mô tối đa 3B tham số. Chưa rõ liệu Mamba có duy trì ưu thế tương đối ở quy mô 7B, 13B hay 70B – ngưỡng phổ biến của các mô hình mã nguồn mở hiện đại.

**② Hệ sinh thái downstream:**
Các kỹ thuật như fine-tuning có hướng dẫn (*instruction tuning*), học tăng cường từ phản hồi con người (RLHF), quantization và prompting với Mamba chưa được nghiên cứu hệ thống.

**③ Đánh đổi continuous–discrete:**
Cơ chế chọn lọc khắc phục điểm yếu trên dữ liệu rời rạc, nhưng có thể ảnh hưởng đến inductive bias của SSM trên dữ liệu liên tục như audio. Các ablation trên waveform audio xác nhận đánh đổi này tồn tại.

**④ Thách thức engineering ở quy mô lớn:**
Scaling SSM lên kích thước lớn hơn có thể đòi hỏi các điều chỉnh engineering bổ sung chưa được thảo luận trong bài báo.

**Hướng nghiên cứu tương lai:**
- Đánh giá Mamba ở quy mô 7B+ và so sánh trực tiếp với LLaMA, Mistral
- Nghiên cứu mô hình hybrid Mamba–Transformer kết hợp ưu điểm hai kiến trúc
- Ứng dụng Mamba trong mô hình hóa video dài và nhận dạng hành động (*action recognition*)
- Triển khai Mamba trên thiết bị biên (*edge devices*) tận dụng ưu thế inference O(1)

---

## SLIDE 20 – Kết luận và ý nghĩa nghiên cứu

**Ba đóng góp kỹ thuật chính:**

| Đóng góp | Nội dung | Tác động |
|---|---|---|
| ① Cơ chế Chọn lọc | Tham số hóa B, C, Δ phụ thuộc đầu vào (S4 → S6) | Khắc phục giới hạn LTI; suy luận nhận biết nội dung |
| ② Thuật toán Hardware-Aware | Kernel fusion + parallel scan + recomputation | Huấn luyện nhanh như FlashAttention, O(L) |
| ③ Kiến trúc Mamba | Block đồng nhất, không attention | Đơn giản, hiệu quả, dễ scale |

**Kết quả nổi bật được xác nhận thực nghiệm:**
- Mô hình linear-time *đầu tiên* đạt chất lượng Transformer trên language modeling
- Throughput suy luận nhanh hơn Transformer cùng kích thước ~5 lần
- Hiệu năng cải thiện liên tục theo context length đến hơn 1 triệu token
- State-of-the-art trên language modeling, audio generation và DNA modeling

**Ý nghĩa đối với hướng nghiên cứu Action Recognition / Edge Device:**
Mamba mở ra hướng triển khai mô hình chuỗi thời gian hiệu quả trên thiết bị biên nhờ ba đặc tính bổ sung lẫn nhau: (1) suy luận O(1) không cần KV cache – phù hợp với RAM hạn chế của thiết bị nhúng; (2) ngữ cảnh thời gian dài không bị giới hạn bởi cửa sổ cố định – quan trọng cho video action recognition; (3) chất lượng nhận biết nội dung tương đương Transformer – đảm bảo độ chính xác trong phân loại hành động.

---

## Tài liệu tham khảo chính

1. **Gu, A., & Dao, T.** (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
2. **Gu, A., Goel, K., & Ré, C.** (2022). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*.
3. **Dao, T., Fu, D. Y., Saab, K., et al.** (2023). Hungry hungry hippos: Towards language modeling with state space models. *ICLR 2023*.
4. **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.
5. **Smith, J. T. H., Warrington, A., & Linderman, S. W.** (2023). Simplified state space layers for sequence modeling. *ICLR 2023*.
6. **Poli, M., et al.** (2023). Hyena hierarchy: Towards larger convolutional language models. *ICML 2023*.

---

*Tài liệu được biên soạn phục vụ báo cáo nghiên cứu khoa học.*
*Cập nhật lần cuối: tháng 07/2026.*
