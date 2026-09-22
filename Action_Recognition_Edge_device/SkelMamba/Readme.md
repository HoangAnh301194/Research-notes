# SkelMamba

- **Bài báo gốc:** `2411.19544v1.pdf`
- **Bản dịch tiếng Việt:** [SkelMamba_translating.md](SkelMamba_translating.md)
- **Chủ đề:** Skeleton-based Action Recognition bằng State Space Model, hướng tới chẩn đoán rối loạn thần kinh và inference hiệu quả.

## Nội dung chính

SkelMamba chia motion feature thành spatial, temporal và spatio-temporal stream. Part-Grouped Mamba tiếp tục chia skeleton theo body part như arms, legs, torso; dùng C-2D-SSM với four-way scanning; sau đó hợp nhất part-level và global feature bằng channel attention.

Mô hình đạt kết quả mạnh trên NTU RGB+D, NTU RGB+D 120, NW-UCLA và ba thiết lập chẩn đoán y khoa; thời gian suy luận được báo cáo là 7,06 ms với 6,84M tham số và 9,7G FLOPs.

## Cách hiểu nhanh các keyword

- **Skeleton:** Không phải ảnh X-ray. Đây là chuỗi tọa độ joint của cơ thể theo từng frame, ví dụ vai, khuỷu tay, đầu gối, cổ chân.
- **Joint / bone:** Joint là một keypoint; bone thường là vector nối hai joint kề nhau, được dùng như một input modality bổ sung.
- **Spatial:** Quan hệ giữa nhiều joint trong cùng một frame, ví dụ tay đang ở gần đầu hay chân đang dang rộng.
- **Temporal:** Sự thay đổi của joint qua nhiều frame, ví dụ quỹ đạo đầu gối trong một chu kỳ gait.
- **Spatio-temporal:** Học đồng thời quan hệ giữa các joint và sự thay đổi của chúng theo thời gian.
- **Anatomically guided / body-part-aware:** Không phải một module phân tích anatomy. Tác giả chỉ group joint theo body part như arms, legs, torso và các tổ hợp arms-legs, arms-torso, torso-legs.
- **Four-way scanning:** Quét tensor skeleton theo spatial-to-temporal, temporal-to-spatial và hai chiều đảo ngược tương ứng.
