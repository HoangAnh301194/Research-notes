# Simba

- **Paper:** `2404.07645v1.pdf`
- **Bản dịch:** [Simba_translating.md](Simba_translating.md)
- **Chủ đề:** Kết hợp Mamba với U-ShiftGCN cho Skeleton Action Recognition (SAR).

## Ý tưởng chính

Mỗi Simba module gồm bốn stage: Down-sampling Shift S-GCN Encoder, Intermediate Mamba Block, Up-sampling Shift S-GCN Decoder và Shift T-GCN. ShiftGCN học spatial structure giữa joint; Mamba học long-range temporal dependency giữa các pose snapshot.

## Cách hiểu keyword

- **Skeleton:** Chuỗi tọa độ joint theo frame, không phải ảnh X-ray.
- **Spatial modeling:** Học quan hệ giữa các joint trong cùng frame.
- **Temporal modeling:** Học sự thay đổi của pose qua nhiều frame.
- **Shift S-GCN:** Graph operation cho spatial graph.
- **Shift T-GCN / ShiftTCN:** Temporal refinement bằng shift operation.
- **Intermediate Mamba:** Mamba nằm giữa encoder và decoder, xử lý sequence các graph snapshot.
- **U-ShiftGCN:** Simba sau khi bỏ Intermediate Mamba Block.
- **Partition gating:** Trộn joint-level feature với body-part/partition-level feature bằng learnable gate.

Simba báo cáo 96,34% trên NW-UCLA với 4-stream ensemble; 89,03%/94,38% trên NTU RGB+D 60; 79,75%/86,28% trên NTU RGB+D 120 với joint modality.
