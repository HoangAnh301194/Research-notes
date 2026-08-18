# Literature review: Mamba đã giải những bài toán action/gesture nào, và được đặt ở đâu trong pipeline?

**Phạm vi thời gian:** 2024–10-08-2026  
**Trọng tâm:** visual temporal action/gesture understanding; ưu tiên application-focused và application + adaptation; không tổ chức kết quả theo “các biến thể Mamba”.

## Tóm tắt

Mamba đã được dùng khá rộng cho **nhận dạng action đã cắt clip**, **skeleton action recognition**, **temporal action localization/detection**, và **action anticipation**; nhưng literature về **continuous/online gesture từ RGB stream** còn hạn chế.

Trong corpus chính gồm **35 công trình thị giác**:

- 17 công trình có ít nhất một thí nghiệm action/activity classification trên clip;
- 5 công trình trực tiếp liên quan gesture/sign (trong đó chỉ 4 là gesture recognition/detection đúng nghĩa, 1 là sign-language translation);
- 10 công trình dùng skeleton/pose/3D geometry ở một phần pipeline;
- 18 công trình có ít nhất một setting ngoài isolated classification: untrimmed/dense labeling, localization, online detection hoặc anticipation;
- chỉ **3 công trình peer-reviewed/chính thức** thỏa tiêu chí strict causal online action detection: **Mamba-OTR, MOAD/Backtrace Mamba, BiOMamba**;
- chỉ **1 công trình trực tiếp về gesture RGB trong video untrimmed** dùng Mamba: *Micro-gesture Online Recognition using Learnable Query Points*; nhưng đó là detector theo **window 128 frame**, không phải suy luận causal giữ trạng thái trên stream;
- số công trình đã xác minh thỏa đồng thời **continuous RGB gesture + causal inference + persistent Mamba state + tự tìm gesture boundary** là **0**.

Kết luận quan trọng nhất không phải “Mamba có bao nhiêu biến thể”, mà là có bốn cách Mamba đang được dùng:

1. làm backbone spatiotemporal cho clip đã cắt;
2. làm temporal encoder trên feature đã trích sẵn;
3. làm bộ nhớ/temporal core causal cho online action detection;
4. làm module fusion, query interaction hoặc diffusion denoiser trong pipeline chuyên biệt.

Literature trưởng thành nhất ở (1) và skeleton recognition; non-causal localization/anticipation đang tăng nhanh; còn streaming gesture đúng nghĩa vẫn là khoảng trống có bằng chứng mạnh.

## 1. Câu hỏi, protocol và tiêu chí

### 1.1 Câu hỏi dẫn đường

Review này bắt đầu từ ba câu hỏi:

1. Bài toán action/gesture cụ thể nào đã được giải bằng Mamba?
2. Input đi qua pipeline nào, và Mamba nằm chính xác ở đâu?
3. Phương pháp có thực sự online/causal/streaming hay chỉ dùng nhãn “online”?

### 1.2 Nguồn và chiến lược tìm kiếm

Đã tra paper gốc/publisher/repository tại arXiv, CVF Open Access, ECCV Open Access, AAAI Proceedings, ACM DL, IEEE Xplore, Springer, ScienceDirect, CEUR-WS và trang repository chính thức. Nhóm từ khóa bao gồm:

- `Mamba action recognition`, `VideoMamba action recognition`, `Mamba skeleton action recognition`;
- `Mamba gesture recognition`, `Mamba hand gesture`, `Mamba dynamic gesture`, `Mamba micro-gesture`;
- `Mamba continuous gesture`, `Mamba gesture spotting`, `Mamba online gesture`, `Mamba streaming gesture`;
- `Mamba temporal action detection/localization/segmentation`;
- `Mamba online action detection`, `causal Mamba video`, `persistent Mamba state video`;
- `Mamba action anticipation`, `Mamba action forecasting`, `egocentric Mamba action`;
- các synonym: trimmed/untrimmed, dense action, frame-wise action, OAD, TAL, TAS, LTA, take/release, sign translation, micro-action.

Citation snowballing được thực hiện từ các paper trực tiếp, phần Related Work, các paper được trích dẫn, paper mới trích dẫn chúng, repository chính thức và challenge reports. Tìm kiếm được xem là bão hòa khi các truy vấn gesture/online/streaming lặp lại cùng các công trình: bài MiGA 2024, MSF-Mamba, ME-GCN, các paper online **action** (không phải gesture), hoặc các phương pháp không dùng Mamba.

### 1.3 Inclusion/exclusion

**Đưa vào corpus chính** khi paper dùng Mamba/Mamba-derived block trong pipeline có input thị giác hoặc biểu diễn bắt nguồn từ thị giác (RGB, event, pose, skeleton, optical flow, point cloud, pre-extracted video feature) và output là action/gesture/segment/boundary/future action.

**Giữ ở phụ lục** nếu là thesis/technical report, sensor-only (EMG/IMU/radar), static image gesture, hoặc thông tin chính chưa xác minh được.

**Không tính là Mamba core** nếu tác giả chỉ nói “Mamba-inspired” nhưng thực chất dùng linear attention khác, ví dụ MGMILA; hoặc dùng S4/S5/S6 chung mà không phát triển trực tiếp từ Mamba.

Không gộp hai version của cùng một công trình. Khi có publication cuối cùng, publication đó được ưu tiên hơn arXiv.

### 1.4 Ba mức đóng góp

- **A — Application-focused:** lấy Mamba/VideoMamba đã có để giải bài toán ứng dụng.
- **B — Application + adaptation:** sửa scan, motion/fusion/memory/head để phù hợp action/gesture.
- **C — Architecture-focused:** kiến trúc Mamba là đóng góp chính; action recognition chủ yếu là benchmark. C chỉ được giữ khi giúp hiểu pipeline.

### 1.5 Định nghĩa temporal setting dùng trong review

| Nhãn | Điều kiện được dùng trong review |
|---|---|
| Isolated/trimmed offline | Clip đã có sẵn start/end; output thường là một class. |
| Untrimmed offline | Toàn bộ video/window có thể dùng cả quá khứ và tương lai; output proposal, boundary hoặc dense label. |
| Temporal segmentation | Gán nhãn mỗi frame/segment trong chuỗi dài; vẫn có thể offline/bidirectional. |
| Strict causal online | Prediction tại thời điểm \(t\) chỉ dùng frame/feature \(\le t\). |
| Streaming | Ngoài causal, model chạy tăng dần và giữ state/memory giữa các bước, không tái xử lý toàn bộ history. |
| Anticipation | Dùng quan sát quá khứ để dự đoán action hoặc chuỗi action tương lai. |

Tên paper có chữ “online” hoặc con số latency thấp **không đủ** để xếp online/streaming.

### 1.6 Ghi chú về venue rank

Rank chỉ được ghi khi xác minh được; dấu “—” không có nghĩa venue kém mà chỉ là review không suy đoán. Snapshot [ICORE 2026](https://portal.core.edu.au/conf-ranks/) xác nhận CVPR/ECCV/AAAI là A* và ICME là A; ICASSP được ghi theo CORE 2023 (B). Không dùng rank để loại paper.

## 2. Census và bản đồ bài toán

### 2.1 Số lượng theo temporal setting chủ đạo

Các nhóm dưới đây **loại trừ nhau theo setting chủ đạo** để tổng bằng 35. Những paper đa nhiệm được tách vào “mixed”.

| Setting chủ đạo | Số paper | Ví dụ đại diện |
|---|---:|---|
| Isolated/trimmed offline | 17 | hai VideoMamba; MSF-Mamba; Simba; ActionMamba; Manta-FSAR; SpikMamba; EventAction |
| Untrimmed offline / dense TAL-TAS-seq2seq | 9 | MiGA query-points; MS-Temba; MambaTAD; SSMamba; SBM; EAV-Mamba; GALFu-Mamba |
| Strict causal online | 3 | Mamba-OTR; MOAD; BiOMamba |
| Anticipation-only | 3 | QueryMamba; MANTA; MixANT |
| Mixed nhiều task/setting | 3 | Video Mamba Suite; Mamba4D; Mamba Fusion/MambaVL |
| **Tổng** | **35** | |

Streaming là tập con của causal online. Mamba-OTR mô tả rõ recurrent-state inference từng frame; MOAD duy trì online hierarchical memory; BiOMamba chỉ quét forward-then-backward trên **past memory có sẵn**, nên causal nhưng không đồng nghĩa dùng recurrent state vô hạn.

### 2.2 Số lượng theo bài toán/modality (không loại trừ nhau)

| Nhóm | Số paper trong corpus chính | Độ trưởng thành |
|---|---:|---|
| RGB/video action classification | 8 | Cao về benchmark clip; thấp về deployment stream end-to-end |
| Skeleton/pose/3D action/gesture | 10 | Cao cho isolated recognition; thấp cho continuous/online |
| Gesture/micro-gesture/sign trực tiếp | 5 | Mỏng; chủ yếu isolated, chỉ 1 windowed detector |
| Temporal detection/localization/segmentation (action hoặc gesture) | 13 | Đang tăng nhanh; phần lớn bidirectional/offline; 3/13 là strict OAD |
| Strict causal online action detection | 3 | Mới hình thành nhưng đã có peer-reviewed top venues |
| Anticipation/forecasting | 6 | Khá đa dạng: causal encoder, query decoder, diffusion |
| Event-camera action/sign | 4 | Có bằng chứng khả thi, nhưng phần lớn vẫn gom event thành clip/frame offline |
| Multimodal action/gesture | 4 | Có RGB+skeleton, RGB+IMU, RGB+text, audio-visual; ít kiểm tra missing modality |

## 3. Master table — task → input → vị trí Mamba → output

Ký hiệu temporal: **I** isolated offline; **U** untrimmed offline; **Sg** temporal segmentation/sequence labeling offline; **C** strict causal online; **St** streaming stateful; **A** anticipation; **Mix** nhiều setting. “NR” = paper không báo cáo hoặc chưa xác minh được. Result là một số đại diện, không dùng để so chéo dataset.

| # | Paper, tác giả, năm | A/B/C | Task; temporal | Modality; input | Base/pipeline và vị trí Mamba | Output | Dataset; result đại diện | Venue; rank | Code; link |
|---:|---|:---:|---|---|---|---|---|---|---|
| 1 | **VideoMamba: State Space Model for Efficient Video Understanding** — Kunchang Li et al., 2024 | C | Clip action recognition; I | RGB; 3D tube/patch tokens | 3D patch embed → bidirectional Mamba backbone → CLS/classifier | Action class | K400, SSv2; paper báo trade-off tốt trên video dài | ECCV 2024; A* | [Code](https://github.com/OpenGVLab/VideoMamba); [paper](https://arxiv.org/abs/2403.06977) |
| 2 | **VideoMamba: Spatio-Temporal Selective State Space Model** — Jinyoung Park et al., 2024 | C | Clip action recognition; I | RGB clip | Spatiotemporal tokens → forward/backward SSM blocks → classifier | Action class | K400 77.7 Top-1 (32f); SSv2 64.2; 26.8M/68G | ECCV 2024; A* | [Code](https://github.com/jinyjelly/VideoMamba); [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03565.pdf) |
| 3 | **Video Mamba Suite** — Guo Chen et al., 2024/2025 | C | Recognition, TAL, TAS, anticipation; Mix | RGB hoặc pre-extracted video features | Mamba thay Transformer trong backbone, ActionFormer, ASFormer và causal TeSTra branch | Class, boundary/dense label, future action | TAL HACS 44.56 avg mAP; EK100 anticipation action R@5 15.2 | IJCV, online 2025; — | [Code](https://github.com/OpenGVLab/video-mamba-suite); [paper](https://arxiv.org/abs/2403.09626); [DOI](https://doi.org/10.1007/s11263-025-02597-y) |
| 4 | **Micro-gesture Online Recognition using Learnable Query Points** — Pengyu Liu et al., 2024 | A | Micro-gesture detection; U, windowed | RGB → I3D RGB features; 128-frame windows | I3D → PointTAD-style decoder; Mamba+MHSA xử lý **query vectors**, không xử lý frame tokens trực tiếp | Start/end/class proposals | SMG cross-subject; F1 14.34, challenge rank 2 | IJCAI MiGA workshop; — | Code NR; [paper](https://ceur-ws.org/Vol-3848/paper_6.pdf) |
| 5 | **MSF-Mamba: Motion-Aware State Fusion Mamba for Efficient Micro-Gesture Recognition** — Deng Li et al., 2026 | B | Isolated micro-gesture; I | RGB; 16 uniformly sampled frames, 224² | 3D patch embed → MSF-Mamba with central-frame-difference motion/local state fusion → classifier | Gesture class | iMiGUE 61.32 (S), 62.98 (M+); SMG 56.22 (S+) Top-1 | IEEE TMM 2026; — | Code NR; [arXiv](https://arxiv.org/abs/2510.10478); [DOI](https://doi.org/10.1109/TMM.2026.3668511) |
| 6 | **Lightweight Dynamic Gesture Recognition based on ShuffleNetV2–Mamba Hybrid (Shuma)** — 2025 | B | Isolated dynamic gesture; I | RGB clips; proprietary/self-built sets | Feature map split → ShuffleNetV2 local branch + sequence Mamba global branch → shuffle/fusion → GAP | Gesture class | Max reported ACC 89.7%; 2.1 MB | FCIS 2025; — | Code NR; [paper](https://doi.org/10.54097/7ms8ar63) |
| 7 | **Mamba-Enhanced GCN for Skeleton-Based Stereoscopic Hand Gesture Recognition in AR (ME-GCN)** — Fucheng Wan, Jian Teng, 2026 | B | Isolated hand gesture; I | 3D skeleton/depth topology | Skeleton → Spatial Graph Mamba temporal scan + depth-aware topology refinement + channel-temporal fusion | Gesture class | SHREC'17 97.38/94.16; DHG 94.28/92.14; 0.68M, 11.52 ms | Computer Animation and Virtual Worlds 2026; — | Code NR; [DOI](https://doi.org/10.1002/cav.70164) |
| 8 | **Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos** — Soumyabrata Chaudhuri, Saumik Bhattacharya, 2024 | B | Skeleton action recognition; I | Joint/bone skeleton streams | downsample Shift S-GCN → temporal Mamba → upsample S-GCN → Shift T-GCN | Action class | NW-UCLA joint 94.18; NTU120 X-sub 89.03 | arXiv; — | Code chưa xác minh; [paper](https://arxiv.org/abs/2404.07645) |
| 9 | **ActionMamba: Action Spatial–Temporal Aggregation Network Based on Mamba and GCN** — Jinglong Wen, Dan Liu, Bin Zheng, 2025 | B | Skeleton action recognition; I | Multi-stream skeleton | SM-GCN + ST-Mamba trong Action Perception Module → classifier | Action class | NTU60/120, UAV-Human; exact best result xem paper | Electronics 2025; — | Code NR; [paper](https://www.mdpi.com/2079-9292/14/18/3610) |
| 10 | **Dual-path Spatio-temporal Mamba for Skeleton-based Action Recognition (SkeMamba)** — Jie Zhao et al., 2025 | B | Skeleton action recognition; I | Skeleton graph → token sequences | Adaptive topology transform → dual-path spatial/temporal Mamba + ST-GateMba | Action class | NTU RGB+D family; exact result NR trong nguồn truy cập | The Visual Computer 2025; — | Code NR; [DOI](https://doi.org/10.1007/s00371-025-03950-5) |
| 11 | **TSkel-Mamba** — Yanan Liu et al., 2025 | B | Skeleton action recognition; I | Skeleton | Spatial Transformer → temporal Mamba + multi-scale temporal interaction | Action class | NTU60/120, NW-UCLA, UAV-Human; SOTA claimed | arXiv; — | Code NR; [paper](https://arxiv.org/abs/2512.11503) |
| 12 | **SkelMamba: A State Space Model for Efficient Skeleton Action Recognition of Neurological Disorders** — Niki Martinel et al., 2024 | B | Skeleton action/clinical movement; I | Skeleton with anatomical partitions | spatial + temporal + spatiotemporal Mamba streams → fusion/classifier | Action/disorder class | NTU60/120, NW-UCLA + clinical dataset; up to +3.2 pp | arXiv; — | Code NR; [paper](https://arxiv.org/abs/2411.19544) |
| 13 | **Manta: A Matryoshka Mamba for Few-Shot Action Recognition** — X. Ma et al., 2025 | B | Few-shot clip action recognition; I | RGB frame features | CNN/ViT/VMamba frame backbone → inner local-subsequence Mamba + outer temporal Mamba/alignment → metric classifier | Episode action class | SSv2 64.7/88.7 (1/5-shot, long setting); supports up to 128 frames | AAAI 2025; A* | Code NR; [paper](https://arxiv.org/abs/2412.07481) |
| 14 | **Mamba4D** — Jiuming Liu et al., 2025 | B | 4D point-cloud action recognition + segmentation; Mix | Point-cloud video | intra-frame spatial Mamba → inter-frame temporal Mamba → task head | Clip class hoặc frame labels | MSR-Action3D +10.4 pp; HOI4D +0.7 F1; 87.5% memory reduction | CVPR 2025; A* | [Paper/code links](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Mamba4D_Efficient_4D_Point_Cloud_Video_Understanding_with_Disentangled_Spatial-Temporal_CVPR_2025_paper.html) |
| 15 | **SpikMamba: When SNN Meets Mamba in Event-based HAR** — 2024 | B | Event action recognition; I | Event stream binned to event tensors | Spiking stem/local window attention + global Mamba → classifier | Action class | PAF/HARDVS/DVS128 Gesture/E-FAction; +1.45/+7.22/+0.15/+3.92 pp | ACM MMAsia 2024; — | Code NR; [DOI](https://doi.org/10.1145/3696409.3700204) |
| 16 | **Event Stream based HAR / EVMamba** — X. Wang et al., 2024/2026 | B | Event action recognition; I | Event stream → spatial planes/voxel sequence | multi-direction spatial Mamba + temporal voxel scan → classifier | Action class | CeleX-HAR; exact table xem paper | arXiv 2024; IJCV 2026 version; — | Code NR; [preprint](https://arxiv.org/abs/2408.09764) |
| 17 | **Event-CSL: Continuous Sign Language Translation with Event Cameras and Mamba** — 2024 | A/B | Continuous sign translation; U/seq2seq offline | Event frames | ResNet-18 per-frame features → Mamba temporal fusion → language decoder | Text/gloss sequence | Event-CSL: 14,827 videos, 2,544-word vocabulary | arXiv; — | Code NR; [paper](https://arxiv.org/abs/2408.10488) |
| 18 | **Mamba-MHAR** — Trung-Hieu Le et al., 2025 | A/B | Multimodal HAR; I | Egocentric RGB + inertial | VideoMamba visual branch + Mamba sensor branch → late fusion | Activity class | UESTC-MMEA-CL, MuWiGes; exact results xem paper | J. Computer Science & Cybernetics 2025; — | Code NR; [paper](https://vjs.ac.vn/jcc/article/view/22770) |
| 19 | **Cross-Modal Action Recognition in Egocentric Video Using Mamba** — Juan I. Bustos Gorostegui, Maria E. Buemi, 2026 | B | Egocentric clip action; I | RGB + hand skeleton | VideoMamba + skeleton-Mamba → one Mamba fusion block/CLS mixing → classifier | Action class | H2O; average CLS mixing +10 pp Tiny, +25 pp Small over reported video-only baseline | arXiv; — | Code NR; [paper](https://arxiv.org/abs/2605.24302) |
| 20 | **Mamba Fusion: Learning Actions Through Questioning (MambaVL)** — Apoorva Beedu et al., 2025 | A/B | Egocentric recognition + 1-s anticipation; Mix | RGB + generated text question | video encoder + RoBERTa text → projections → shared-state Mamba fusion → classifier | Verb/noun/action class hoặc future action | EK100; recognition action 55.0; anticipation action 23.9 Top-1 | ICASSP 2025; B (CORE2023) | [Code](https://github.com/dongzhikang/mambavl); [DOI](https://doi.org/10.1109/ICASSP49660.2025.10888933) |
| 21 | **QueryMamba** — Zeyun Zhong et al., 2024 | A | Long-term action anticipation; A | Egocentric RGB → VideoMAE features | long/short feature memories → Mamba encoder → 20-query Transformer decoder + verb-noun co-occurrence rerank | 20 future verb-noun pairs | Ego4D LTA; 2nd challenge, action edit 0.8663 | CVPR Ego4D challenge report; — | [Code](https://github.com/zeyun-zhong/querymamba); [paper](https://arxiv.org/abs/2407.04184) |
| 22 | **MANTA: Diffusion Mamba for Stochastic Long-Term Dense Action Anticipation** — Olga Zatsarynna et al., 2025 | B | Dense stochastic anticipation; A | I3D/TSM observed features + zero-padded future/noise | 15-block bidirectional Mamba diffusion denoiser | Multiple future frame-wise action/duration sequences | Breakfast/50Salads/Assembly101; 1.4M params, 1.1 s/25 samples; 65.3× inference vs GTDA | CVPR 2025; A* | [Code](https://github.com/olga-zats/DIFF_MANTA); [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zatsarynna_MANTA_Diffusion_Mamba_for_Efficient_and_Effective_Stochastic_Long-Term_Dense_CVPR_2025_paper.html) |
| 23 | **MixANT** — 2025 | B/C | Dense stochastic anticipation; A | Observed I3D/TSM features + diffusion latent | early vanilla Mamba → bidirectional MoE MixMamba diffusion blocks | Multiple future action sequences | Breakfast/50Salads/Assembly101; paper reports consistent SOTA | ICCV 2025; A* | Code NR; [paper](https://arxiv.org/abs/2509.11394) |
| 24 | **Mamba-OTR** — Alessandro S. Catinello et al., 2025/2026 | A | Online take/release endpoint detection; C+St | Pre-extracted egocentric visual features at 4 fps | projection → 3 vanilla Mamba layers → per-frame head; recurrent Mamba state carried frame-by-frame | take/release/background endpoint | EK100 subset; 45.48 sliding-window, 43.35 streaming mp-mAP | Springer chapter 2026 / arXiv; — | Code announced; [paper](https://arxiv.org/abs/2507.16342); [DOI](https://doi.org/10.1007/978-3-032-10185-3_36) |
| 25 | **Backtrace Mamba / MOAD** — Shiyu Yan et al., 2026 | B | Online action detection; C+stateful memory | Pre-extracted RGB+flow features | Mamba → hierarchical compressed memory bank + causal motion trigger/pruning → second Mamba → current-frame classifier | Current action/background label | THUMOS14 74.3 mAP; TVSeries 91.2 mcAP; FineAction 41.2 mAP; 34 FPS | AAAI 2026; A* | Code NR; [paper](https://ojs.aaai.org/index.php/AAAI/article/view/38139); [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38139/42101) |
| 26 | **BiOMamba** — Sensen Wang, Yuehu Liu, Chi Zhang, 2025 | B | Online action detection + anticipation; C | Pre-extracted short-/long-term past memory | compress distant past + keep recent past → forward-then-backward Mamba **trên history đã thấy** → OAD/OAA heads | Current và future action | THUMOS OAD 73.3, OAA 59.7 mAP; TVSeries 89.9/83.7 mcAP | ACM MM 2025; — | Code NR; [DOI](https://doi.org/10.1145/3746027.3755847) |
| 27 | **MS-Temba** — Arkaprava Sinha et al., 2026 | B | Dense multi-label TAD in long ADL videos; U | Frozen I3D/CLIP segment features | multi-dilation bidirectional Temba blocks → Mamba multi-scale fuser → per-time classifier | Dense action labels/boundaries | TSU/Charades 44.0/33.6 mAP with CLIP; 17M, 3.46G, 51 samples/s | CVPR 2026; A* | [Code](https://github.com/thearkaprava/MS-Temba); [paper](https://arxiv.org/abs/2501.06138) |
| 28 | **MambaTAD** — Hui Lu et al., 2026 | B | One-stage temporal action detection; U | Raw/video backbone or pre-extracted features, tùy setting | state-space temporal adapter + diagonal-masked **bidirectional** SSM pyramid → global fusion detection head | Start/end/class proposals | HACS 44.9 avg mAP; FineAction 29.4; MultiTHUMOS 35.9 | IEEE TMM 2026; — | Code NR; [paper](https://arxiv.org/abs/2511.17929); [DOI](https://doi.org/10.1109/TMM.2026.3676839) |
| 29 | **Temporal Action Localization with State-Sensitive Mamba and Centroid Sequences Enhancement (SSMamba)** — Peng Wang et al., 2025 | B | Two-stage TAL; U | Pre-extracted video features | proposal generation/refinement with state-sensitive Mamba + centroid sequence enhancement | Proposal start/end/class | THUMOS/ActivityNet family; exact result xem paper | Neurocomputing 620:129246; — | Code NR; [DOI](https://doi.org/10.1016/j.neucom.2024.129246) |
| 30 | **Enhanced TAL with Separated Bidirectional Mamba and Boundary Correction (SBM)** — Xiangbin Liu, Qian Peng, 2025 | B | TAL; U | Pre-extracted video features | forward Mamba ∥ backward Mamba → pre-localization → frame-contribution boundary correction | Start/end/class | THUMOS13 73.7; ActivityNet 42.0; HACS 45.2; FineAction 29.1 mAP | Mathematics 2025; — | Code NR; [paper](https://www.mdpi.com/2227-7390/13/15/2458) |
| 31 | **Transformer or Mamba for TAL?** — Zejian Zhang, Cristina Palmero, Sergio Escalera, 2025 | A | Controlled TAL encoder comparison; U | Pre-extracted video features | same TAL pipeline, thay Transformer encoder bằng bidirectional Mamba variants | Temporal proposals | THUMOS14/ActivityNet; kết quả phụ thuộc block/feature | VISAPP 2025; rank chưa xác minh hiện hành | Code NR; [paper](https://www.scitepress.org/Papers/2025/131730/131730.pdf); [DOI](https://doi.org/10.5220/0013173000003912) |
| 32 | **EAV-Mamba** — Quan Zhang et al., 2025 | B | Weakly supervised TAL; U | Audio + visual features | modality encoders → efficient audio-visual Mamba representation/fusion → WS-TAL head | Action segments from video-level labels | Standard WS-TAL benchmarks; exact result NR trong nguồn mở | ICME 2025; A | Code NR; [DOI](https://doi.org/10.1109/ICME59968.2025.11210145) |
| 33 | **GALFu-Mamba** — Shuaibiao Zhang et al., 2025 | B | Skeleton temporal action segmentation; Sg | Long skeleton sequence | local joint/temporal modeling + global Mamba fusion → frame classifier/refinement | Frame-wise action labels | Skeleton TAS benchmarks; exact table NR | IEEE SmartWorld 2025; — | Code NR; [DOI](https://doi.org/10.1109/SWC65939.2025.00171) |
| 34 | **ETMamba: An Effective Temporal Model for Video Action Recognition** — R. Hong et al., 2026 | C | Clip action recognition; I | RGB video tokens/features | efficient temporal Mamba architecture → classifier | Action class | Public action datasets; exact table xem paper | Electronics 2026; — | Code NR; [paper](https://www.mdpi.com/2079-9292/15/6/1338) |
| 35 | **EventAction: Vision Mamba-Based Event-Driven Action Recognition** — Qingyu Wang et al., 2026 | B | Event-pose action recognition; I | Event frames → pose heatmaps, 48-frame groups | Efficient-VMamba+MamLSTM pose estimator → MambaAction (VMamba + inverted 3D conv) | Action class | CDEHP; pose AP 81.68; action accuracy cao nhất trong comparison; pose stage 44 FPS | VISAPP 2026; — | Code NR; [paper](https://doi.org/10.5220/0014489100004084) |

### 3.1 Ma trận thông tin bổ sung cho 35 paper

Bảng này bổ sung các field không thể đặt gọn trong master table: pretraining/initialization, thành phần được thay thế, baseline, efficiency, lý do dùng Mamba và hạn chế. “Chưa xác minh” được giữ nguyên thay vì suy đoán.

| # | Pretraining / Mamba thay gì | Baseline hoặc đối chứng đáng chú ý | Efficiency được báo cáo | Vì sao chọn Mamba | Hạn chế liên quan câu hỏi review |
|---:|---|---|---|---|---|
| 1 | Image/video pretraining theo variant; Mamba thay self-attention trong backbone | 3D CNN, ViT/video Transformer | Linear theo token length; nhiều cấu hình dài | Global dependency với memory thấp hơn attention | Bidirectional, clip-level; không boundary/background/stateful stream |
| 2 | ImageNet-style initialization; pure Mamba thay ViT blocks | Video Transformer và CNN | 26.4–26.8M; 34–68 GFLOPs | Pure SSM cho spatial-temporal tokens | Uniform fixed clips; dùng future frames trong clip |
| 3 | Dùng backbone/features pretrained tùy task | ActionFormer, ASFormer, TeSTra và Transformer tương ứng | Giảm memory/parameters ở nhiều task | Kiểm tra Mamba như drop-in temporal alternative | Nhiều task dùng feature trích sẵn; không phải một deployed system thống nhất |
| 4 | I3D RGB pretrained Kinetics-400; Mamba trong query decoder | PointTAD-like decoder; 2025 DyFADet comparator không dùng Mamba | NR; 2 Mamba blocks tốt nhất, sâu hơn gặp gradient explosion | Long-range interaction giữa learnable query points | Window 128 frame; test window reset; không causal/persistent; F1 thấp |
| 5 | Theo VideoMamba setting; exact checkpoint tùy variant | VideoMamba-S: 58.13 iMiGUE/53.28 SMG | S: 33.38M; S+: 97.28M | Motion rất nhỏ cần local state fusion + long context | 16 frame uniformly sampled; Top-1 chỉ khoảng 60%; không spot boundary |
| 6 | Chưa xác minh | CNN/ShuffleNet variants | 2.1 MB; paper nói giảm compute 43.6% | Ghép local CNN và global sequence modeling | Self-built datasets; không protocol streaming/latency end-to-end |
| 7 | Chưa xác minh | TD-GCN và skeleton-GCN khác | 0.68M, 1.08 GFLOPs, 11.52 ms | Long temporal dependency trên joint graph, phù hợp AR edge | Isolated clips; tác giả nêu continuous streaming là future work |
| 8 | Skeleton model thường train task-specific; exact init NR | U-ShiftGCN, GCN/Transformer SAR | NR trong nguồn chính đã trích | Mamba làm temporal core giữa các spatial GCN | Offline clip; không boundary/background; preprint |
| 9 | Chưa xác minh | GCN và skeleton Transformer | Chưa xác minh | Long-range joint/time aggregation | Bidirectional/offline; không continuous |
| 10 | Chưa xác minh | GCN/Mamba SAR | Chưa xác minh | Joint spatial-temporal union context | Offline; exact runtime và streaming chưa báo cáo |
| 11 | Chưa xác minh | Transformer-Mamba ablations | Low inference time được claim; số exact NR | Mamba temporal, Transformer spatial | Preprint; online inference chỉ là future direction |
| 12 | Chưa xác minh | Skeleton SOTA và Transformer | Lower complexity được claim; exact NR | Anatomical multistream long dynamics | Clip classification; clinical generalization còn hẹp |
| 13 | ImageNet-pretrained ResNet/ViT/VMamba frame encoder | FSAR Transformer methods | 4.25–4.61 h/10k tasks trên RTX3090; hỗ trợ 128f | Nested Mamba tránh attention OOM với long subsequences | Few-shot episodes vẫn dùng trimmed clips; không dense/online |
| 14 | Task-specific 4D point initialization; exact NR | 4D point Transformer/point baselines | 87.5% memory reduction; 5.36× speedup ở long sequences | Tách spatial state và temporal state | Point-cloud domain; segmentation offline, không RGB streaming |
| 15 | Event-task training; exact pretrain NR | SNN/event HAR SOTA | Improvement được báo cáo; absolute FPS NR | SNN local dynamics + Mamba global dependency | Event tensor clip offline; “low latency sensor” không chứng minh causal model |
| 16 | Event-task pretraining/training; exact NR | Event CNN/Transformer | NR | Spatial-plane và voxel temporal scans | Offline classification; không giữ state qua arbitrary stream |
| 17 | ImageNet-pretrained ResNet-18 được nêu | Event sign translation baselines | NR | Gộp temporal view/event sequence hiệu quả | Offline sentence decoding; không gesture spotting/latency |
| 18 | VideoMamba + Mamba sensor models; exact checkpoints NR | Unimodal và multimodal HAR | Nhẹ được claim; exact table xem paper | Hai temporal modalities đều phù hợp SSM | Clip classification; sensor availability/domain-specific |
| 19 | VideoMamba pretrained SSv2; skeleton encoder train trên H2O | Video-only VideoMamba và four CLS strategies | Chỉ thêm 1 Mamba fusion block | Linear multimodal token fusion; preserve pretrained CLS sinks | 8-frame clips; small dataset; không online dù motivation nhắc online |
| 20 | Pretrained ORViT/ViT/AVION + RoBERTa | ORViT; Transformer fusion | 413G/157M vs Transformer fusion 413.5G/242M | Shared state cho vision-language fusion | Question do GPT-4o/recognized action tạo offline; error propagation; clip-level |
| 21 | VideoMAE Kinetics-pretrained rồi Ego4D-finetuned | Ego4D LTA challenge systems | NR | Encode 64 s long + 30 s short history tuyến tính | Decoder vẫn là Transformer; technical report; không current-action streaming |
| 22 | Pre-extracted I3D/TSM features | GTDA diffusion | 1.4M, 10.2 GB, 1.1 s/25 samples; 65.3× faster inference | Long joint past-future latent with linear complexity | Bidirectional denoiser; requires complete observed prefix; not online detector |
| 23 | Same feature families as MANTA | MANTA và anticipation SOTA | Exact table xem paper | MoE state dynamics cho nhiều plausible futures | Architecture-heavy; offline diffusion; not causal gesture |
| 24 | Pre-extracted visual features; Mamba trained on 20-frame clips | Vanilla Transformer 20.32; vanilla Mamba 25.16 mp-mAP | 0.14 s/video; paper also reports 8 ns/frame excluding extractor—không nên coi là E2E latency | Recurrent inference without reprocessing history | Feature extractor excluded; only take/release endpoints; abstract/table vs intro có số không nhất quán |
| 25 | Pre-extracted RGB+flow with standard pretrained backbones | MiniROAD/LSTR/Transformer OAD; base Mamba | 10.2G, 34 FPS, 41.2 FineAction mAP | Recover critical distant context with compressed memory | Not raw-video E2E; memory compression/PCA/quantization complexity |
| 26 | Pre-extracted feature memory | OAD/OAA Transformer methods | Latency/FPS chưa xác minh từ full paper | Forward then backward over **past only** enriches current representation | Bounded/compressed memory; exact state persistence and raw E2E cost NR |
| 27 | Frozen I3D or CLIP visual backbone | MS-TCT: 87M, 27.4G, 22.2 samples/s | 17M, 3.46G, 51.0 samples/s | Dilated SSM captures multi-scale long ADL actions | Bidirectional and whole-video padded length; offline, backbone cost excluded |
| 28 | VideoMAE/VideoMAEv2/InternVideo features or E2E adapter | ActionFormer, TriDet, DyFADet, AdaTAD | 9.3–16.7G backbone setting; detailed table in paper | Long-span context + global fusion at linear temporal cost | Deliberately uses future context; not causal/streaming |
| 29 | Pre-extracted feature setting | Two-stage TAL baselines | Chưa xác minh | Reduce state decay; enhance proposal centroid context | Offline two-stage; precise future access/runtime NR in accessible text |
| 30 | Pre-extracted feature setting | Contemporary TAL SOTA | Runtime/FPS NR | Forward/backward state + explicit boundary correction | Bidirectional; no causal deployment; THUMOS is reported as “THUMOS13” in paper |
| 31 | Same features/head across controlled variants | Transformer, DBM/ViM/Mamba blocks | Lower complexity is central comparison; exact depends config | Isolate whether Mamba can replace attention in TAL | Experimental comparison, not a streaming application; bidirectional variants |
| 32 | Audio/visual pretrained features; exact init NR | WS-TAL audio-visual baselines | Efficient claimed; exact public table unavailable | Long cross-modal temporal context from weak labels | Video-level supervision, offline; no causal protocol |
| 33 | Skeleton task training; exact init NR | Skeleton TAS GCN/TCN baselines | Chưa xác minh | Local joints + global temporal sequence | Whole-sequence frame labels; causality/state reset not reported |
| 34 | Chưa xác minh | CNN/Transformer/video SSM methods | Chưa xác minh | Efficient temporal action modeling | Architecture-focused; clip benchmark, no boundary/streaming |
| 35 | E-VMamba backbone; two stages trained separately | tDenseRNN pose + SimPoseC3D | Pose 44 FPS, 23.35M/118.69G; MambaAction 9.05M/53.49G | Event global context + pose temporal modeling | 48-frame aggregation; authors explicitly say offline/non-real-time; pose dominates latency |

## 4. Phân loại theo **bài toán ứng dụng**, không theo tên Mamba

| Nhóm bài toán | Paper chính | Mamba được dùng như thế nào | Đánh giá maturity |
|---|---|---|---|
| **A. Isolated RGB action recognition** | Hai VideoMamba; Manta-FSAR; ETMamba | Thường là 3D patch/tube embedding → bidirectional Mamba backbone → classifier; hoặc temporal Mamba trên frame features | **Cao về benchmark**. Có pretraining và scale studies, nhưng gần như không kiểm tra background, boundary, false alarm hay state carry. |
| **B. Isolated RGB gesture recognition** | MSF-Mamba; Shuma; luận văn *Micro-gesture recognition using Mamba* | VideoMamba/motion-aware state fusion hoặc CNN local branch ∥ Mamba global branch | **Thấp–trung bình**. Ít paper, dataset/protocol không đồng nhất; Shuma dùng self-built data; chưa có strong continuous benchmark. |
| **C. Skeleton/pose action recognition** | Simba; ActionMamba; SkeMamba; TSkel-Mamba; SkelMamba; EventAction stage 2 | GCN/Transformer spatial encoder → Mamba temporal; hoặc tokenize joint graph rồi scan spatial/temporal nhiều hướng | **Khá trưởng thành cho isolated clips**. Nhiều biến thể và benchmark NTU, nhưng almost no causal continuous evaluation. |
| **D. Skeleton/pose gesture recognition** | ME-GCN | Skeleton → graph-Mamba temporal scan + topology refinement → class | **Mỏng**. Một paper trực tiếp, kết quả isolated tốt; chính tác giả để continuous streaming là future work. |
| **E. Multimodal action/gesture** | Mamba-MHAR; Cross-Modal RGB+hand skeleton; MambaVL; EAV-Mamba | Separate Mamba encoders rồi late/CLS/state fusion; Mamba có thể là cross-modal fusion core | **Đang hình thành**. Có gain từ fusion, nhưng robustness khi thiếu modality, sync drift và streaming fusion chưa được đánh giá đầy đủ. |
| **F. Micro-gesture recognition** | MSF-Mamba; MiGA learnable query points; luận văn 2025 | Motion-aware VideoMamba cho clip; hoặc Mamba-MHSA ở decoder queries cho proposal | **Mỏng**. Paper tốt nhất vẫn clip classification; detector MiGA là windowed offline và F1 thấp. |
| **G. Continuous gesture recognition** | Event-CSL (continuous sign translation, event camera); MiGA 2024 (gần liên quan) | Mamba temporal fusion cho sequence-to-text; query decoder cho start/end/class | **Chưa trưởng thành**, đặc biệt với RGB. Không có paper đã xác minh thỏa RGB + causal + stateful + arbitrary stream. |
| **H. Gesture spotting** | MiGA 2024 là gần nhất | I3D features → learnable query points → Mamba/MHSA decoder → temporal proposals | **Một điểm dữ liệu**. Có boundary, nhưng future frames trong window và không persistent state. |
| **I. Temporal action detection/localization** | Video Mamba Suite/ActionMamba; MS-Temba; MambaTAD; SSMamba; SBM; comparison VISAPP; EAV-Mamba | Mamba thay temporal Transformer/feature pyramid, làm bidirectional long-range encoder/fuser; detection head dự đoán proposal/boundary | **Tăng nhanh và khá mạnh**, nhưng phần lớn intentionally bidirectional/offline. |
| **J. Temporal action segmentation** | ASMamba trong Video Mamba Suite; Mamba4D; GALFu-Mamba | Mamba thay self-attention trong ASFormer hoặc temporal core cho point/skeleton frame labels | **Có bằng chứng nhưng ít hơn TAL**. Hầu hết whole-sequence offline; chưa có causal TAS Mamba được xác minh. |
| **K. Strict causal online action detection** | Mamba-OTR; MOAD; BiOMamba | Pre-extracted feature stream → causal Mamba/past-memory → current-frame label; Mamba-OTR giữ recurrent state, MOAD giữ hierarchical memory | **Mới nhưng rõ ràng nhất cho deployment logic**. Ba paper cho thấy Mamba có lợi, nhưng chưa end-to-end từ raw camera. |
| **L. Online gesture recognition** | Không có paper thỏa strict definition; MiGA chỉ mang tên “online” | — | **Gap mạnh**. Cần tách rõ tên challenge và computational protocol. |
| **M. Streaming action/gesture** | Mamba-OTR; MOAD; MADAM thesis (phụ lục) | Incremental Mamba state hoặc online compressed memory; không re-run full history | **Rất ít**. Chỉ action; không có general gesture vocabulary từ RGB stream. |
| **N. Early action recognition/anticipation** | Video Mamba Suite/TeSTra; QueryMamba; MambaVL; MANTA; MixANT; BiOMamba | Causal short-memory encoder, long/short Mamba + query decoder, hoặc bidirectional diffusion over observed+latent future | **Khá đa dạng và phát triển nhanh**. Tuy nhiên anticipation không tự động là streaming: nhiều model cần cả observed prefix rồi chạy batch. |
| **O. Human–robot interaction/assembly** | MADAM thesis; ME-GCN AR; Mamba-MHAR (motivation) | Streaming temporal core hoặc lightweight skeleton/Mamba fusion | **Application evidence còn yếu**. Chỉ MADAM trực tiếp đặt trong assembly/HRC và đó là thesis; chưa có closed-loop robot latency/safety study. |

### 4.1 Hai nhầm lẫn cần tránh

1. **Action localization offline không phải online detection.** MambaTAD, SBM và MS-Temba mạnh vì dùng cả context trước/sau. Thiết kế bidirectional chính là lý do chúng không thể được chuyển nhãn thành causal chỉ vì Mamba gốc có recurrence.
2. **Action anticipation không đồng nghĩa online inference.** MANTA và MixANT dự báo tương lai, nhưng denoiser có thể chạy batch trên observed prefix và latent future; MambaVL lấy một clip quan sát cố định. Chúng dự đoán future nhưng không chứng minh stateful stream.

## 5. Các paper quan trọng nhất khi nhìn từ pipeline

### 5.1 MiGA 2024: Mamba ở **query decoder**, không phải video backbone

**Paper:** Pengyu Liu, Fei Wang, Kun Li, Guoliang Chen, Yanyan Wei, Shengeng Tang, Zhiliang Wu, Dan Guo, *Micro-gesture Online Recognition using Learnable Query Points*, IJCAI 2024 MiGA Workshop ([PDF](https://ceur-ws.org/Vol-3848/paper_6.pdf)).

**Pipeline được xác minh:**  
RGB frames → I3D RGB-only pretrained Kinetics-400 → temporal features (stride 4, spatial average pool) → PointTAD-style action decoder → Mamba-MHSA xử lý learnable queries → FFN → \((start,end,class)\).

**Temporal verdict:** inference dùng non-overlapping window 128 frame sau khi training với overlap 0.75. Model không báo cáo causal mask, không giữ Mamba state giữa window và dùng toàn bộ context trong window. Vì vậy đây là **untrimmed offline/windowed detection**, không phải strict online.

**Kết quả/giới hạn:** SMG cross-subject, RGB-only, F1 14.34; 2 Mamba blocks tốt nhất, sâu hơn gặp gradient explosion. Không có latency/FPS. Bài MiGA 2025 không dùng Mamba đạt F1 38.03 bằng VideoMAEv2-g + DyFADet + spatial-temporal attention, cho thấy “có Mamba” chưa phải yếu tố quyết định của challenge ([comparator 2025](https://ceur-ws.org/Vol-4168/paper_3.pdf)).

### 5.2 MSF-Mamba: motion-aware clip classifier, không tự spot gesture

**Paper:** Deng Li et al., *MSF-Mamba: Motion-Aware State Fusion Mamba for Efficient Micro-Gesture Recognition*, IEEE TMM 2026 ([DOI](https://doi.org/10.1109/TMM.2026.3668511)).

**Pipeline:** 16 RGB frames sampled đều → 3D patch embedding → stacks of MSF-Mamba/MSF-Mamba+ → central-frame-difference motion cue được trộn vào local state → pooling/classifier.

**Mamba thay gì:** thay global spatiotemporal self-attention của video backbone, nhưng được bổ sung explicit motion-local-state module vì vanilla VideoMamba dễ để static appearance lấn át subtle motion.

**Output/metric:** một class/clip; iMiGUE 61.32 Top-1 với S và 62.98 với M+; SMG 56.22 với S+. VideoMamba-S baseline là 58.13/53.28. S model 33.38M parameters; S+ 97.28M.

**Temporal verdict/limit:** isolated offline; không background/no-gesture, không start/end, không multiple gestures. Authors ghi nhận confusion giữa các gesture thị giác gần giống và đề xuất body-parts/keypoints.

### 5.3 ME-GCN: latency thấp cho clip skeleton không chứng minh stream

**Paper:** Fucheng Wan, Jian Teng, *Mamba-Enhanced GCN for Skeleton-Based Stereoscopic Hand Gesture Recognition in Augmented Reality*, 2026 ([DOI](https://doi.org/10.1002/cav.70164)).

**Pipeline:** 3D skeleton → Spatial Graph Mamba Module → stereoscopic depth-aware topology refinement → channel-temporal dual-stream fusion → class.

**Kết quả:** SHREC'17 14/28 class 97.38/94.16%; DHG-14/28 94.28/92.14%; 0.68M parameters, 1.08 GFLOPs, trung bình 11.52 ms.

**Verdict:** isolated skeleton gesture. Chính phần future work nêu mở rộng sang continuous gesture recognition trong streaming AR; do đó 11.52 ms là latency per isolated sample/module, không phải bằng chứng đã xử lý endless stream.

### 5.4 Mamba-OTR: bằng chứng rõ nhất cho recurrent streaming

**Paper:** Alessandro Sebastiano Catinello, Giovanni Maria Farinella, Antonino Furnari, *Mamba-OTR: a Mamba-based Solution for Online Take and Release Detection from Untrimmed Egocentric Video* ([arXiv](https://arxiv.org/abs/2507.16342), [Springer DOI](https://doi.org/10.1007/978-3-032-10185-3_36)).

**Pipeline:** visual feature đã trích ở 4 fps → linear projection → 3 vanilla Mamba layers → current-frame take/release/background head. Training dùng clip 20 frame (5 s), focal loss và fixed-window regularizer để tránh nhiều peak quanh một endpoint. Inference truyền từng frame và giữ recurrent Mamba state, không cần buffer/reprocess history.

**Output:** endpoint hiện tại của “take”, “release” hoặc background—không phải full temporal segment.

**Kết quả:** vanilla Transformer 20.32, vanilla Mamba 25.16, final 45.48 mp-mAP ở sliding-window và 43.35 ở streaming. Paper báo 0.14 s/video và 8 ns/frame khi loại feature extraction; con số ns không nên diễn giải là end-to-end camera latency. Intro cũng có một con số 51.76 không khớp abstract/table; review lấy bảng/abstract 45.48/43.35.

**Giá trị:** đây là paper gần nhất với pipeline `stream features → persistent Mamba state → online prediction`, nhưng vocabulary chỉ có hai event thao tác và backbone visual vẫn offline/pre-extracted.

### 5.5 MOAD/Backtrace Mamba: causal memory cho long-form OAD

**Paper:** Shiyu Yan et al., *Backtrace Mamba: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection*, AAAI 2026 ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/38139)).

**Pipeline:** pre-extracted RGB+optical-flow features → projection/position → first Mamba → PCA/quantized hierarchical scene/action memory bank cập nhật online bằng similarity → causal motion-aware trigger + temporal soft pruning → second Mamba → classifier cho current frames.

**Temporal verdict:** strict causal; paper định nghĩa OAD không dùng future frames và memory bank được cập nhật online. Đây là stateful system ở cấp memory, dù không phải raw-camera end-to-end.

**Kết quả:** THUMOS14 72.4/74.3 mAP tùy feature pretraining; TVSeries 89.8/91.2 mcAP; FineAction 41.2 mAP; EK100 mean recall@5 overall verb/noun/action 47.1/50.5/28.4. Trên FineAction, full model 10.2 GFLOPs, 34 FPS, 41.2 mAP so với base 17.9 GFLOPs, 25 FPS, 32.4 mAP.

**Giới hạn:** feature extractor và optical-flow cost nằm ngoài temporal model; hierarchical memory cần PCA/quantization/update logic; chưa đánh giá gesture onset/offset hoặc false positive per hour.

### 5.6 BiOMamba: “backward” nhưng vẫn causal vì chỉ quét **past memory**

**Paper:** Sensen Wang, Yuehu Liu, Chi Zhang, *BiOMamba: Mamba-based Forward-Then-Backward Temporal Modeling for Online Action Detection and Anticipation*, ACM MM 2025 ([DOI](https://doi.org/10.1145/3746027.3755847)).

**Pipeline:** available history → compress distant long-term memory, retain recent short-term memory → forward Mamba → backward Mamba trên đúng memory đó → representation cho current action và future action heads.

**Temporal verdict:** causal nếu backward pass không chứa frame \(>t\). Bidirectional **trên quá khứ** khác với bidirectional **trên cả video** của MambaTAD/SBM. Full paper không truy cập được trong lần rà này nên backbone, latency và chi tiết state carry được ghi NR.

**Kết quả đã xác minh từ publisher/accepted-paper record:** THUMOS14 OAD 73.3 mAP, OAA 59.7; TVSeries OAD 89.9 mcAP, OAA 83.7.

### 5.7 MS-Temba và MambaTAD: Mamba là temporal detector mạnh, nhưng chủ động non-causal

**MS-Temba** ([CVPR 2026 paper](https://arxiv.org/abs/2501.06138)) dùng frozen I3D/CLIP segment features → Temba blocks với nhiều dilation, mỗi branch là bidirectional SSM → auxiliary scale losses → SSM multi-scale fuser → per-timestep multi-label classifier. Nó đạt 44.0/33.6 mAP trên TSU/Charades với CLIP, 17M parameters, 3.46 GFLOPs và 51 samples/s. Padded temporal length có thể tới 2,500 segment. Vì bidirectional scan và whole-video padding, đây là dense offline TAD, không phải stream.

**MambaTAD** ([IEEE TMM 2026](https://doi.org/10.1109/TMM.2026.3676839)) dùng state-space temporal adapter, Diagonal-Masked Bidirectional SSM ở multi-scale feature pyramid và global feature fusion head. Paper nói rõ causal vanilla Mamba làm mất future context cần cho boundary, rồi thiết kế DMBSS để dùng cả trước và sau. Nó đạt 44.9 avg mAP trên HACS và 29.4 trên FineAction. Đây là bằng chứng rất rõ rằng linear temporal complexity và online causality là hai thuộc tính khác nhau.

### 5.8 Anticipation: Mamba có ba vai trò khác nhau

1. **Causal short-memory replacement:** Video Mamba Suite thay short causal attention trong TeSTra bằng vanilla Mamba; trên EK100 5-second features, overall verb/noun/action recall@5 là 27.9/34.1/15.2, so với 25.1/30.8/14.1 của short attention.
2. **Long/short encoder + query decoder:** QueryMamba dùng Mamba để encode 64 s long memory và 30 s short memory, sau đó Transformer decoder với 20 queries dự đoán 20 verb–noun pairs. Đây là hybrid, không phải “Mamba-only”.
3. **Diffusion denoiser:** MANTA/MixANT dùng bidirectional Mamba để denoise toàn bộ chuỗi latent past+future. Không có future visual frame; “future” là variable cần sinh. Tuy nhiên inference vẫn là batch stochastic anticipation, không phải current-frame streaming detection.

MANTA đáng chú ý về efficiency: 1.4M parameters, 10.2 GB, 1.1 s/video cho 25 samples, so với GTDA 3.9M, 19.2 GB, 71.8 s; tức 65.3× nhanh hơn theo setup paper ([CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zatsarynna_MANTA_Diffusion_Mamba_for_Efficient_and_Effective_Stochastic_Long-Term_Dense_CVPR_2025_paper.html)).

## 6. Các mẫu ứng dụng Mamba rút ra từ corpus

| Pattern | Pipeline chuẩn hóa | Bài toán dùng nhiều | Paper đại diện | Nhận xét |
|---|---|---|---|---|
| **P1. End-to-end video backbone** | RGB clip → 3D patch/tube embedding → bidirectional Mamba → classifier | Isolated action/micro-gesture | hai VideoMamba, MSF-Mamba, ETMamba | Phổ biến nhất cho clip. Mạnh về long tokens, nhưng bidirectional/fixed clip. |
| **P2. Temporal Mamba trên feature trích sẵn** | CNN/ViT/I3D/CLIP → temporal feature sequence → Mamba → head | TAL/TAD/OAD/anticipation | Suite, MS-Temba, MambaTAD, Mamba-OTR, MOAD | Phổ biến nhất trong untrimmed tasks. Efficiency thường chỉ đo temporal head, không gồm backbone. |
| **P3. Mamba trong detection/query head** | video features → learnable queries → Mamba/MHSA → proposals | Gesture spotting/TAD | MiGA 2024 | Ít gặp; cho thấy “Mamba ở đâu” quan trọng hơn tên model. |
| **P4. GCN/pose + temporal Mamba** | skeleton/pose → GCN/spatial encoder → Mamba temporal → classifier | Skeleton action/gesture | Simba, ME-GCN, ActionMamba | Rất phổ biến; spatial inductive bias vẫn do GCN/graph topology đảm nhiệm. |
| **P5. Multimodal state/token fusion** | modality encoders → align/project → Mamba fusion/shared state → class | RGB+skeleton, RGB+text, audio-visual | Cross-Modal H2O, MambaVL, EAV-Mamba | Gain có thật nhưng chưa rõ khi modality bị thiếu hoặc lệch thời gian. |
| **P6. Persistent causal core** | streaming features → recurrent Mamba state / online memory → current prediction | Strict OAD | Mamba-OTR, MOAD, MADAM thesis | Mẫu phù hợp nhất với deployment; ít paper nhất và chưa có direct RGB gesture. |
| **P7. Hybrid CNN/attention + Mamba** | local CNN/attention ∥ global Mamba → fusion | Lightweight gesture/event action | Shuma, SpikMamba, EventAction | Mamba không thay mọi thứ; local motion/spatial bias vẫn cần module riêng. |
| **P8. Explicit motion + Mamba** | appearance tokens + frame-difference/flow/motion trigger → Mamba | Micro-gesture/OAD | MSF-Mamba, MOAD | Quan trọng khi state update dễ bị static background chi phối. |
| **P9. Diffusion Mamba** | observed features + noisy future latent → bidirectional Mamba denoiser | Stochastic dense anticipation | MANTA, MixANT | Tận dụng long sequence tốt, nhưng hoàn toàn khác online recurrence. |
| **P10. Spatial–temporal scan decomposition** | spatial scan per frame → temporal scan across frames | Point cloud/event/skeleton | Mamba4D, EVMamba, SkeMamba | Giải quyết việc flatten token tùy ý làm mất topology; thường architecture-heavy. |

### 6.1 Pattern nào thống trị từng task?

- **Isolated action:** P1 và P4.
- **Gesture:** P1/P7/P8; corpus nhỏ, chưa có consensus.
- **Continuous/offline:** P2, thường bidirectional và pre-extracted features.
- **Strict online:** P6, đôi khi kết hợp P8; chỉ ba paper chính.
- **Anticipation:** P2 encoder hoặc P9 diffusion.

## 7. Bảng kiểm offline–online–streaming

“Future frame” nói về input visual tại prediction time, không phải latent future trong diffusion. “Persistent state” chỉ ghi Có khi paper mô tả state/memory được mang qua bước thời gian; không suy từ công thức Mamba.

| Paper | Task | Dùng future visual frame? | Sliding/window? | Persistent state/memory? | Tự tìm boundary? | Real-time evidence? | Kết luận temporal |
|---|---|---|---|---|---|---|---|
| VideoMamba (2 papers) | Clip class | Có, trong clip/bidirectional | Fixed clip | Không báo cáo | Không | Throughput/FLOPs, không stream | Isolated offline |
| MSF-Mamba | Micro-gesture class | Có, 16-frame clip | Fixed clip | Không | Không | Params/FLOPs, không latency stream | Isolated offline |
| Shuma | Dynamic gesture class | Có trong clip | Fixed clips | Không báo cáo | Không | Lightweight claim; E2E FPS NR | Isolated offline |
| ME-GCN | Skeleton hand gesture | Có trong full sequence | Fixed isolated sequence | Không | Không | 11.52 ms/sample, không stream | Isolated offline |
| MiGA query-points | Micro-gesture proposals | Có trong 128-frame window | Có; test non-overlap | Không; reset/không báo state carry | Có, trong window | Không | Untrimmed windowed offline |
| Event-CSL | Sign translation | Có trong sentence sequence | Sequence batch | Không báo cáo | Không phải spotting | Không | Offline seq2seq |
| Video Mamba Suite—TAL/TAS | TAL/TAS | Có với ViM/DBM/ASFormer | Whole video/features | Không | TAL: Có; TAS: dense labels | Efficiency, không causal | Untrimmed offline |
| Video Mamba Suite—anticipation | Future action | Không | Fixed observed memory | Causal Mamba; state carry NR | Không | NR | Anticipation, causal encoder |
| Mamba4D segmentation | 4D action segmentation | Có trong full point sequence | Sequence batch | Không báo cáo | Dense frame labels | Speed/memory, không causal | Offline segmentation |
| MANTA / MixANT | Dense future sequence | Không future **visual**; có latent future | Whole observed prefix + diffusion steps | Không | Sinh duration/labels, không detect current boundary | 1.1 s/25 samples (MANTA) | Offline anticipation |
| QueryMamba | 20 future actions | Không | 64s+30s memories | Không báo recurrent carry | Không | NR | Offline/challenge anticipation |
| MambaVL anticipation | Future verb/noun/action | Không | 16-frame observed clip | Không | Không | GFLOPs, không stream | Clip anticipation |
| MS-Temba | Dense multi-label TAD | Có; bidirectional | Whole/padded sequence | Không | Dense temporal labels | 51 samples/s temporal system | Untrimmed offline |
| MambaTAD | TAD proposals | Có; design chủ động bidirectional | Whole feature sequence | Không | Có | Complexity table, không stream | Untrimmed offline |
| SSMamba | TAL | Chưa xác minh chi tiết; model offline | Feature sequence | Không báo cáo | Có | NR | Untrimmed offline |
| SBM | TAL | Có; forward+backward | Whole feature sequence | Không | Có + boundary correction | NR | Untrimmed offline |
| EAV-Mamba | Weakly supervised TAL | Chưa xác minh; offline | Whole video features | Không báo cáo | Có từ weak labels | NR | Untrimmed offline |
| GALFu-Mamba | Skeleton TAS | Chưa xác minh; whole sequence | Long sequence | Không báo cáo | Dense labels | NR | Offline segmentation |
| **Mamba-OTR** | Online take/release endpoint | **Không** | Training clips; inference framewise; also SW ablation | **Có: recurrent Mamba state** | Endpoint event, không full span | Temporal head timing; extractor excluded | **Strict causal streaming** |
| **MOAD** | Online action detection | **Không** | Online chunks/features | **Có: hierarchical online memory** | Current frame class, không explicit segment end | **34 FPS temporal system** | **Strict causal/stateful** |
| **BiOMamba** | OAD + OAA | **Không**; backward chỉ trên past | Short/long past memory | Past-memory buffer Có; recurrent state exact NR | Không explicit start/end | FPS/latency NR | **Strict causal** |
| EventAction | Event action class | Có trong 48-frame group | 4-frame pose clips → 48-frame action clip | MamLSTM trong clip; cross-clip carry không báo cáo | Không | Pose 44 FPS; authors nói non-real-time overall | Isolated offline |

### 7.1 Kết luận từ bảng

Trong 22 dòng đại diện, chỉ ba dòng cuối nhóm online action không dùng future visual frame. Trong số đó, chỉ Mamba-OTR mô tả đúng “Mamba hidden state mang từ frame này sang frame khác” một cách không nhập nhằng; MOAD mang external/compressed memory; BiOMamba mang bounded past memory. Vì vậy không nên dùng cụm “Mamba vốn recurrent nên mọi Mamba video đều stream được”.

## 8. Khoảng trống nghiên cứu — chỉ kết luận sau khi đã rà corpus

### 8.1 Câu hỏi đích: hiện có bao nhiêu paper thực sự làm continuous/online RGB gesture với Mamba?

Theo tiêu chí đồng thời:

1. input là **RGB video stream**;
2. có background/no-gesture và nhiều gesture theo thời gian;
3. prediction tại \(t\) không dùng future frame;
4. Mamba state hoặc temporal memory được giữ qua các bước;
5. model tự spot onset/offset hoặc phát hiện current gesture;

**số paper đã xác minh trong corpus là 0**.

Nếu nới từng điều kiện:

- **RGB + gesture + tự tìm boundary nhưng không causal/stateful:** 1 — MiGA 2024 learnable query points.
- **RGB + isolated micro-gesture, không boundary/background:** 2 peer-reviewed/journal-style — MSF-Mamba và Shuma; cộng 1 thesis.
- **Skeleton + isolated hand gesture:** 1 — ME-GCN.
- **Event camera + continuous sign sequence, offline seq2seq:** 1 — Event-CSL.
- **Causal/stateful Mamba nhưng là action, không phải gesture:** 3 — Mamba-OTR, MOAD, BiOMamba; cộng MADAM thesis.

Đây là kết luận **strong evidence trong phạm vi snapshot 10-08-2026**, không phải tuyên bố tuyệt đối rằng không thể tồn tại một unpublished/internal system.

### 8.2 Gap matrix

| Possible gap | Paper gần nhất đã làm được gì | Chưa giải quyết gì | Mức chắc chắn |
|---|---|---|---|
| **Continuous RGB gesture + causal + persistent state** | MiGA 2024 có RGB và boundary; Mamba-OTR có causal recurrent state; MSF-Mamba có gesture motion-aware features | Chưa paper nào ghép đủ ba năng lực; MiGA nhìn cả window, Mamba-OTR chỉ take/release action, MSF cần clip cắt sẵn | **Strong** |
| **Arbitrary-length gesture stream và state lifecycle** | Mamba-OTR chạy framewise; MOAD có hierarchical memory | Chưa kiểm tra gesture vocabulary, state reset sau scene cut/person change, drift hàng giờ, truncated-training vs indefinite inference | **Strong** |
| **Background/no-gesture, multiple consecutive/overlapping gestures** | MiGA có background/proposals; Mamba-OTR có background endpoint class; MS-Temba xử lý overlapping ADL actions | Không direct gesture paper causal nào đo false positives, missed onset, overlapping/multi-label gesture | **Strong** |
| **Causal boundary/onset prediction** | Mamba-OTR phát endpoint event; MiGA dự đoán start/end offline | Chưa có causal onset+offset head cho general gestures; chưa đánh giá detection delay theo milliseconds/frames | **Strong** |
| **Multi-person, track-aware temporal memory** | Skeleton papers mô hình một sequence; các TAD paper thường dùng global video features | Không có verified online gesture model gắn state riêng theo person/track, xử lý enter/exit và identity switch | **Moderate–strong** |
| **Raw-camera end-to-end streaming** | Mamba-OTR/MOAD/BiOMamba cho temporal core tốt; VideoMamba là raw-clip backbone | Online papers dùng pre-extracted features; chi phí decoder, optical flow, pose và acquisition không nằm trong latency | **Strong** |
| **Edge deployment có measurement chuẩn** | ME-GCN 0.68M/11.52 ms; MS-Temba 17M/3.46G; EventAction báo 44 FPS pose; Shuma 2.1 MB | Không có continuous RGB gesture trên edge với end-to-end FPS, energy, peak memory, warm-up, state size | **Strong** |
| **Closed-loop HRI** | MADAM đặt OAD trong assembly; ME-GCN nhắm AR; Mamba-MHAR nêu HRI | Chưa có study robot closed-loop đo response delay, unsafe false trigger, recovery/cancel gesture, operator variation | **Moderate–strong** |
| **Missing/noisy modalities trong fusion** | Mamba-MHAR, Cross-Modal H2O, MambaVL và EAV-Mamba chứng minh fusion có lợi | Hầu như không có modality dropout, asynchrony, missing skeleton/IMU, degraded camera hoặc calibration drift | **Moderate** |
| **Open-set/unknown gesture và domain shift** | MSF-Mamba/GMoT cho thấy confusion/cross-domain khó; skeleton papers benchmark cross-subject/view | Không có online unknown-rejection, calibration, continual enrollment hoặc OOD false-trigger protocol cho Mamba gesture | **Moderate–strong** |
| **Training–inference state consistency** | Mamba-OTR train 20-frame clips nhưng infer recurrently; đây là proof of concept tốt | Chưa có systematic study về detach/truncated BPTT, carried state during training, state leakage giữa videos, reset policy | **Moderate** |
| **Chuẩn đánh giá streaming gesture** | OAD dùng mAP/mcAP; MiGA dùng F1; isolated dùng Top-1 | Thiếu benchmark thống nhất gồm event-level F1/mAP, onset latency, time-to-detect, false alarms/hour, compute including backbone | **Strong** |

### 8.3 Vì sao không thể gọi gap chỉ từ keyword search?

Review đã kiểm tra thêm ba lớp gần liên quan:

- paper ghi “online” nhưng protocol windowed (MiGA 2024);
- paper ghi “real-time/lightweight” nhưng task isolated (Shuma, ME-GCN, EventAction);
- paper causal/streaming thật nhưng task là action endpoint/OAD (Mamba-OTR, MOAD, BiOMamba).

Các truy vấn `continuous gesture`, `gesture spotting`, `online gesture`, `streaming gesture`, `micro-action`, `hand gesture`, `sign`, cùng citation snowballing đều quay về các lớp trên hoặc các phương pháp không dùng Mamba. Vì vậy gap được đặt ở **giao của các yêu cầu**, không phải ở từ khóa “gesture”.

## 9. Research agenda thực dụng cho continuous RGB gesture

Một hướng nghiên cứu có đóng góp rõ nên bắt đầu từ task/protocol:

### 9.1 Task formulation

Input là RGB stream không giới hạn độ dài; tại mỗi frame/segment \(t\), model phát:

- `background` hoặc class gesture hiện tại;
- onset/offset confidence;
- optional person/track id;
- uncertainty/unknown score.

Tất cả output tại \(t\) chỉ dùng \(x_{\le t}\). State phải có policy initialize/update/reset rõ ràng.

### 9.2 Pipeline tối thiểu có thể kiểm chứng

RGB frames → lightweight causal spatial encoder → per-frame/short-tube features → motion-difference gate → **unidirectional Mamba với state carry** → dual heads `{current class, onset/offset}` → hysteresis/debounce.

Nếu multi-person:

RGB → detector/tracker → ROI feature per track → state table `{track_id: Mamba state}` → per-track gesture/boundary heads.

Điểm novelty nên nằm ở state/memory và streaming loss, không chỉ đặt tên một Mamba block mới.

### 9.3 Baselines bắt buộc

- causal TCN;
- GRU/LSTM cùng parameter budget;
- causal Transformer/TeSTra-style short memory;
- vanilla unidirectional Mamba;
- offline bidirectional upper bound;
- sliding-window version reset state;
- stateful version giữ state.

### 9.4 Ablation bắt buộc

- future-free causal vs bidirectional upper bound;
- state carry vs reset mỗi window;
- motion gate vs không motion;
- train with carried state vs independent clips;
- state size/context compression;
- RGB-only vs pose/skeleton privileged training;
- reset on scene cut/person exit;
- raw backbone included vs temporal-head-only timing.

### 9.5 Metrics phù hợp deployment

Ngoài event mAP/F1 và class accuracy, cần:

- onset/offset error và time-to-detect;
- false positives/hour trong background dài;
- missed gesture rate và duplicate trigger rate;
- performance theo stream length và state age;
- end-to-end FPS, median/p95 latency, peak memory, energy;
- state bytes/person và cost khi số track tăng;
- accuracy khi frame drop, blur, occlusion, illumination/domain shift.

## 10. Phụ lục kỹ thuật và ranh giới corpus

Năm record dưới đây có liên quan nhưng không được dùng để làm phồng census thị giác peer-reviewed chính; cùng 35 paper chính tạo thành **40 record Mamba-related đã catalog**.

| # | Record | Vì sao ở phụ lục | Thông tin hữu ích |
|---:|---|---|---|
| 36 | **Micro-gesture recognition using Mamba** — Partha Durbar Hore, MSc thesis, 2025 ([PDF](https://lutpub.lut.fi/bitstream/handle/10024/170090/mastersthesis_Partha_Durbar_Hore.pdf?isAllowed=y&sequence=1)) | Thesis; full PDF bị access challenge trong lần rà | Dùng VideoMamba cho feature extraction/classification; exact pipeline/result chưa xác minh |
| 37 | **MADAM: A Multimodal Mamba-Based Approach for Online Action Detection in Assembly Scenarios** — Giovanni Cinel, MSc thesis 2025/26 ([record](https://thesis.unipd.it/handle/20.500.12608/110132)) | Thesis, không peer-reviewed | RGB+flow hoặc skeleton → fusion → stacked Mamba thay LSTR temporal Transformer; strict streaming. ATTACH frame mAP 37.28; THUMOS 66.13 vs LSTR 69.50 |
| 38 | **Mamba-based multi-modal driver action recognition** — J. Guo, thesis 2025 ([record](https://dr.ntu.edu.sg/entities/publication/58dd20a8-c205-4486-80ae-179015817af4)) | Thesis; full result chưa xác minh | VideoMamba + cross-modal/hybrid fusion cho driver action |
| 39 | **MV-GMN: Multi-View Graph Mamba Network** — 2025 ([arXiv](https://arxiv.org/abs/2501.13829)) | Preprint và numerical claims không nhất quán giữa snippets; giữ ngoài count chính | Skeleton-guided RGB crop + DeiT/skeleton encoders → cross-attention → graph Mamba qua view/time → class |
| 40 | **ActivityMamba: A CNN-Mamba Hybrid Neural Network for Efficient HAR** — Fei Luo et al., IEEE TMC 2025 ([DOI](https://doi.org/10.1109/TMC.2025.3544573)) | Nhiều sensing representations, không trực tiếp video action/gesture stream trong câu hỏi đích | Hierarchical CNN + visual-Mamba SE blocks; 5 datasets/3 sensing techniques; useful evidence cho efficient HAR |

### 10.1 Sensor-only Mamba — không tính vào visual corpus

- **HARMamba**: wearable IMU HAR; bidirectional Mamba, PAMAP2/WISDM/UNIMIB/UCI ([paper](https://arxiv.org/abs/2403.20183)).
- **EMamba/MoEMba**: EMG gesture recognition; relevant với gesture semantics nhưng không phải visual stream.
- **MMA/Momentum Mamba**: inertial HAR ([paper](https://arxiv.org/abs/2511.21550)).
- **RadMamba**: radar HAR ([paper](https://arxiv.org/abs/2504.12039)).

Chúng cho thấy SSM hợp với sensor sequence, nhưng không thể được dùng làm bằng chứng rằng continuous **RGB** gesture đã được giải.

### 10.2 “Mamba-inspired” — ghi riêng, không tính core

- **MGMILA** dùng *Mamba-inspired Linear Attention (MILA)* cho micro-gesture; đây không phải Mamba state-space block nên không gộp với corpus.
- **GMoT** (ACM MM 2026 preprint) là motion-aware tokenization cho MLLM, không dùng Mamba trong pipeline chính; nó đạt iMiGUE/SMG 67.32/73.11 Top-1 và là comparator quan trọng cho direct gesture progress ([paper](https://arxiv.org/abs/2607.16322)).
- **Online Micro-gesture Recognition Using Data Augmentation and Spatial-Temporal Attention** (MiGA 2025) dùng VideoMAEv2-g + DyFADet + attention, không Mamba ([paper](https://ceur-ws.org/Vol-4168/paper_3.pdf)).

### 10.3 Static gesture — ngoài temporal question

Các bài nhận dạng static ASL alphabet image bằng vision Mamba không được tính, vì không có temporal sequence, boundary, state hoặc streaming gesture.

## 11. Threats to validity và quality control

1. Snapshot dừng ở 10-08-2026; preprint có thể đổi title/venue/result.
2. Một số publisher pages không cho truy cập full text. Khi không xác minh được pretraining, runtime, exact table hoặc code, báo cáo ghi NR thay vì suy đoán.
3. Các kết quả không so chéo trực tiếp vì dataset, split, feature backbone và metric khác nhau. Ví dụ mAP TAL không tương đương mp-mAP endpoint hay Top-1 clip.
4. Latency thường chỉ đo temporal head. Review không gọi “real-time end-to-end” nếu feature extraction/pose/optical flow bị loại khỏi timing.
5. Venue rank là snapshot riêng và không dùng để quyết định inclusion.
6. “SOTA” chỉ được lặp lại như claim của paper khi exact setup chưa thể cross-check; master table ưu tiên số đo cụ thể đã đọc được.

## 12. Kết luận cuối cùng

### 12.1 Con số

- **35 paper trong corpus thị giác chính**; **40 Mamba-related records** nếu cộng 5 technical/peripheral records.
- Theo task không loại trừ nhau: 17 action classification; 5 gesture/sign trực tiếp; 10 skeleton/pose/3D; 18 paper có ít nhất một setting continuous/untrimmed/online/anticipation; 3 strict causal online action papers.
- Theo temporal setting chủ đạo loại trừ nhau: 17 isolated offline; 9 untrimmed offline/dense; 3 strict causal online; 3 anticipation-only; 3 mixed.
- Streaming recurrent/external-memory được mô tả rõ trong 2 paper chính (Mamba-OTR, MOAD); BiOMamba causal qua bounded past memory; MADAM bổ sung một thesis streaming.

### 12.2 Paper gần nhất với gesture

1. **MSF-Mamba** — trực tiếp nhất cho RGB micro-gesture classification và motion-aware adaptation.
2. **Micro-gesture Online Recognition using Learnable Query Points** — trực tiếp nhất cho RGB gesture boundary/proposals, nhưng windowed offline.
3. **ME-GCN** — trực tiếp nhất cho skeleton hand gesture/AR, nhưng isolated.
4. **Shuma** — lightweight RGB dynamic gesture, evidence thấp hơn do self-built datasets.

### 12.3 Paper gần nhất với online action

1. **Mamba-OTR** — recurrent state được giữ frame-by-frame; gần streaming nhất.
2. **MOAD/Backtrace Mamba** — causal OAD với compressed hierarchical memory và FPS được báo cáo.
3. **BiOMamba** — joint OAD/OAA với forward-then-backward modeling trên past-only memory.
4. **MADAM** — assembly/HRI streaming, nhưng mới là thesis.

### 12.4 Câu trả lời cho bài toán đích

Không có paper đã xác minh nào trong snapshot này giải trọn:

> continuous RGB gesture recognition  
> + strict causal inference  
> + persistent temporal state  
> + automatic boundary/background handling.

Các mảnh ghép đã tồn tại riêng rẽ: MSF-Mamba giải subtle gesture representation; MiGA query-points giải proposal/boundary trong window; Mamba-OTR/MOAD/BiOMamba chứng minh causal state/memory cho online action. Nghiên cứu có giá trị nhất tiếp theo là kết hợp các mảnh này trong một **protocol streaming gesture đúng nghĩa**, với raw-camera end-to-end timing và false-trigger metrics, thay vì chỉ đề xuất thêm một biến thể Mamba cho clip classification.
