# TSkel-Mamba

- **Paper:** `2512.11503v1.pdf`
- **Bản dịch:** [TSkel-Mamba_translating.md](TSkel-Mamba_translating.md)
- **Chủ đề:** Hybrid Spatial Transformer–Mamba cho Skeleton-based Action Recognition.

## Ý tưởng chính

TSkel-Mamba tách rõ hai nhiệm vụ: Spatial Transformer học joint dependency trong từng frame; Temporal Dynamics Modeling (TDM) dùng Mamba để học temporal dependency. Multi-scale Temporal Interaction (MTI) bổ sung cross-channel temporal interaction mà vanilla Mamba còn yếu.

## Cách hiểu keyword

- **TDM:** Temporal plugin gồm channel projection, MTI, forward/backward Mamba và temporal pooling.
- **MTI:** Dùng Cycle-FC với nhiều temporal kernel để trộn feature giữa channel và adjacent frame.
- **Temporal-prioritized scanning:** Với mỗi joint, quét toàn bộ frame theo đúng temporal order.
- **RPE:** Relative Position Encoding lấy từ shortest-path distance giữa joint trên skeleton graph.
- **CPKD:** Covariance Pooling + Knowledge Distillation; chỉ tăng cost khi training, không tăng inference cost của student.

TSkel-Mamba báo cáo 87,9% trên NTU120 X-Set, 87,4% trên NTU120 X-Sub và 47,2% trên UAV-Human; 2,4M parameter, 8,2G FLOPs.
