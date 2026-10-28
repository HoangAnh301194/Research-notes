# Truy nguyên nguồn gốc notebook GitHub ↔ Kaggle version (28/10/2026)

**Nguồn kiểm chứng:** [GitHub Actions run #40648689083](https://github.com/HoangAnh301194/Research-notes/actions/runs/40648689083), workflow `Kaggle Notebook Manager`, script `version_match.py`.

## Phương pháp và giới hạn

- Đọc **12 file `.ipynb`** đang có dưới `Research_Report/`; phân tích Kaggle inventory **14 notebook**; kiểm tra sâu **8 notebook ứng viên**, tải được **30 version đánh số**.
- So sánh SHA-256 của chuỗi code cell sau khi chuẩn hóa xuống dòng và khoảng trắng đầu/cuối, độc lập với tên file, metadata notebook và execution count.
- Mức **exact-code**: 100% nội dung và thứ tự code cell trùng. Không chứng minh output, checkpoint, metadata hoặc kết quả đánh giá trùng.
- Mức **cell-overlap**: tỷ lệ tương đồng trên danh sách code-cell hashes (theo SequenceMatcher); không phải tỷ lệ byte hoặc độ tương đồng toàn bộ code.
- Khi hai version có code giống hệt, không thể xác nhận một version riêng biệt nếu không có thêm output / lịch sử xuất file.
- Kaggle CLI `kernels pull` trả mã nguồn version nhưng **không trả output cell** trong các lần kiểm toán này: bên GitHub có output, Kaggle nguồn tải về có 0 output cell. Không thể dùng cách so sánh output trực tiếp trong phương pháp hiện tại.
- **K13 và K14** là alias riêng tư, tương ứng với hai Kaggle notebook khác nhau; không phải tên/slug. Alias được gán theo thứ tự từ `sorted(kaggle kernels list -m)` (K13 là ref thứ 13, K14 là ref thứ 14; danh sách có thể đổi nếu tài khoản thêm/xóa/đổi slug). Không đưa tên riêng tư vào log repository public.

## Kết quả từng file

| File dưới `Research_Report/` | Kaggle alias | Version khớp | Mức khớp | Nhận xét |
|---|---|---|---|---|
| `KaggleNotebook/videomamba-vs-videomaev1-v2.ipynb` | K14 | **v1** | exact-code | Khớp riêng v1 (v2 cũng giống một phần 93,33%). |
| `KaggleNotebook/videomamba_man_configsBenchmark.ipynb` | K14 | **v3 / v4** | exact-code | Không phân biệt v3 và v4 bằng code. |
| `261007/videomamba_man_configsBenchmark.ipynb` | K14 | **v3 / v4** | exact-code | Bản sao tương đương code với file trên. |
| `261007/train_failed_echoch29.ipynb` | K14 | **v6 / v7** | exact-code | V5 đạt 97,30% overlap; chưa tách v6/v7 bằng output. |
| `261007/evaluation.ipynb` | K14 | **v8** | exact-code | Khớp riêng v8. |
| `KaggleNotebook/VideoMambaTinyCTPGestureFinal.ipynb` | K14 | **v12** | exact-code | V11 giống 92,86% code cells. |
| `261015/VideoMambaTinyCTPGestureFinal.ipynb` | K14 | **v12** | exact-code | Bản sao tương đương code với file trên. |
| `KaggleNotebook/EgoEvGesture_VideoMamba_CUDA_Wheel_Builder_Kaggle_EN_v1.ipynb` | K13 | **v1 / v2** | exact-code | Cả hai version có code giống nhau. |
| `KaggleNotebook/videomamba_ctpgesture_v1_train.ipynb` | K14 | **v5 gần nhất** | 74,29% cell-overlap | **Không có version khớp toàn bộ code trong dải quét**. |
| `KaggleNotebook/EgoEvGesture_Notebooks/EgoEvGesture_Heterogeneous_Clean_Split_Publish_Kaggle_EN.ipynb` | — | — | Chưa tìm được | Chưa chứng minh có bản tương ứng ở Kaggle. |
| `KaggleNotebook/EgoEvGesture_Notebooks/EgoEvGesture_Heterogeneous_Published_Split_Validator_v2_PathFix_EN.ipynb` | — | — | Chưa tìm được | Chưa chứng minh có bản tương ứng ở Kaggle. |
| `KaggleNotebook/EgoEvGesture_Notebooks/EgoEvGesture_VideoMamba_Tiny_Train_Resume_v1_EN.ipynb` | — | — | Chưa tìm được | Chưa chứng minh có bản tương ứng ở Kaggle. |

## Những gì đã chứng minh và chưa chứng minh

- **8/12 file GitHub có ít nhất một version Kaggle khớp 100% code.** Một file gần nhất 74,29%; 3 file chưa có ứng viên khớp.
- Thực nghiệm IPN benchmark và CTP train/evaluation không nhất thiết nằm trong các notebook Kaggle có cùng tên ở hiện tại: **K14 đã dùng nhiều version cho nhiều công đoạn nghiên cứu khác nhau**.
- Đây là đối chiếu **mã nguồn**, **không khẳng định** rằng một phiên bản đã tạo `checkpoint-best.pth` hay các con số accuracy trong báo cáo. Việc đó cần so sánh output của version và checksum của checkpoint.
- Đợt quét đã duyệt các version đánh số liên tiếp của 8 notebook ứng viên, dừng sau 4 version không tìm thấy; **không bảo đảm bao phủ các version sau khoảng trống lớn hoặc các notebook không được chọn làm ứng viên**.
- Tên/slug Kaggle riêng tư chưa được ghi vào repo public. Để lấy bảng `owner/slug/version` hoàn chỉnh, phải chuyển thông tin đó qua kênh riêng tư hoặc có sự đồng ý công khai slug.

## Công cụ

`Research_Report/KaggleNotebook/_kaggle_manager/version_match.py` — kiểm toán chỉ đọc Kaggle, SHA code cell, mức overlap, và mức version phù hợp.
