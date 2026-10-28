# Kaggle Notebook Provenance Audit — CTPGesture / IPN-Hand (2026-10-28)

## Phạm vi và phương pháp

- Đã kiểm kê **14 notebook** trong tài khoản Kaggle đã xác thực bằng `KAGGLE_API_TOKEN`.
- GitHub Actions tải **mã nguồn version mới nhất** của từng notebook vào runner tạm thời để nhận diện; kết quả run [#40644987622](https://github.com/HoangAnh301194/Research-notes/actions/runs/40644987622) **success**, 0 lỗi pull/parse.
- **Không in tên/slug/path riêng tư ra log của repository public**, không tải mã nguồn Kaggle vào git, không thực thi GPU hoặc chỉnh sửa Kaggle.
- Chưa lấy được `current_version_number` bằng Kaggle CLI/SDK; danh sách `kernels list` chỉ trả `ref,title,author,lastRunTime,totalVotes`.

## Đối chiếu vai trò

| Nhóm | Đối chiếu GitHub public | Bằng chứng | Số candidate Kaggle latest |
| --- | --- | --- | --- |
| CTP train | `Research_Report/KaggleNotebook/videomamba_ctpgesture_v1_train.ipynb` | `train_one_epoch`, optimizer; notebook public source còn có official single-view và 12-view test | 1 |
| CTP isolated/continuous test | `Research_Report/KaggleNotebook/VideoMambaTinyCTPGestureFinal.ipynb` | `isolated_predictions.csv`, `continuous_raw_cache.pkl` và full test split | 1 |
| IPN train | **Chưa có file nguồn gốc được xác minh trên GitHub** | Có 1 Kaggle latest notebook chứa thành phần VideoMamba + IPN train | 1 |
| IPN isolated/continuous test | `Research_Report/KaggleNotebook/videomamba_man_configsBenchmark.ipynb` | Kaggle candidate có isolated+continuous+VideoMAE+model.eval+benchmark+checkpoint, **chưa xác minh đồng nhất chính xác với file GitHub** | 1 ứng viên mạnh |
| IPN khác | Chưa định danh | 1 notebook Kaggle có IPN/VideoMamba, checkpoint, classification nhưng chưa khớp các protocol trên | 1 chưa phân loại |

## Các bản được lưu và kết quả thực nghiệm từ repo

- `Research_Report/261007/train_failed_echoch29.ipynb` là bản huấn luyện CTP có output tới epoch 29, không mặc nhiên là bản chuẩn hiện tại.
- Báo cáo `261007/Readme.md` ghi best checkpoint ở **epoch 18** (đếm từ 1) và quá trình train bị ngắt ở epoch 29.
- `261015/Readme.md` ghi checkpoint-best epoch **17** (có thể dùng cách đánh số từ 0); chưa khóa cùng hash/checkpoint.
- `Research_Report/261022/P1_NOTEBOOK_RUNBOOK_20261027.md` chỉ ra:
  - CTP test isolated: cell 16; continuous: cell 22.
  - IPN test isolated: cell 12; continuous: cell 18.
- **Chưa thể gắn các kết quả cuối với một Kaggle version number cụ thể**: cần metadata version và output/checkpoint tương ứng.

## Quy tắc nghiệm thu tiếp theo

1. Xác định đúng Kaggle `ref` và current version number **qua kênh không công khai dữ liệu private**.
2. Đối với CTP train: xác định version đã sinh `checkpoint-best.pth`, đối chiếu SHA-256, epoch, split và class mapping.
3. Đối với IPN train: xác định version sinh checkpoint đang được notebook IPN test nạp.
4. Đối với CTP/IPN test: xác định version đã xuất CSV/cache metrics, đối chiếu output với hai báo cáo `261007`/`261015` và runbook `261022`.
5. Không gộp log/metric ở các version khác nhau thành cùng một thí nghiệm; không tải notebook private lên repo public.

**Lưu ý:** Phân loại theo code version *mới nhất* chỉ chứng minh notebook hiện nay có chức năng gì, không chứng minh version cũ nào thực sự tạo ra kết quả đã công bố.
