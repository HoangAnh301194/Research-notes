# Kaggle Notebook Manager (thử nghiệm, chỉ đọc)

Công cụ này nằm trong repository **Research-notes**, giữ nguyên các notebook và báo cáo hiện có.

## Trạng thái / phạm vi

- GitHub Actions cài Kaggle CLI và sử dụng **repository secret** `KAGGLE_API_TOKEN`.
- `check`: kiểm tra token thực sự có quyền gọi Kaggle API; không in tên notebook.
- `inventory`: kiểm kê notebook theo trang, tạo `notebooks.csv` và `summary.json`.
- `pull`: tải **mã nguồn + kernel-metadata.json** của một notebook, hỗ trợ chỉ định số phiên bản Kaggle.
- Không có `push`, không chạy notebook, không download checkpoint/output và không xóa dữ liệu.
- Dữ liệu chỉ tồn tại tạm trên GitHub Actions runner. **Không tự commit vào repo.**

## Chạy trên GitHub

1. Mở [Actions](https://github.com/HoangAnh301194/Research-notes/actions) → **Kaggle Notebook Manager** → **Run workflow**.
2. Chọn `check` trước, để trống notebook/version, giữ **Export files as artifact = false**.
3. Nếu thành công, chọn `inventory`. Lúc này job chỉ in **tổng số** notebook vào log.
4. Để tải một notebook có thể công khai, chọn `pull`, điền `owner/notebook-slug`, và tùy chọn `version` (ví dụ `3`).
5. Chỉ bật **Export files as artifact** nếu cậu đồng ý **toàn bộ dữ liệu xuất ra được xem như công khai**.

### Cảnh báo về repository PUBLIC

Repository này là **public**. Người có quyền đọc repository có thể tải GitHub Actions artifacts; dữ liệu notebook riêng tư, thông tin thí nghiệm chưa công bố, hoặc danh sách tên notebook riêng tư **không được xuất thành artifact**. Vì vậy, tùy chọn xuất artifact mặc định là **false**. Không nhập secret trong notebook hoặc metadata.

Nếu muốn dùng GitHub để lưu **toàn bộ** notebook Kaggle kể cả notebook riêng tư, cần đổi visibility repository sang private hoặc tạo kho private riêng. Không thể bảo vệ notebook riêng tư chỉ bằng một branch trong repository public.

## Kết quả

- `inventory`: `notebooks.csv`, `summary.json`.
- `pull`: thư mục `notebook/` và `pull_manifest.json` ghi SHA-256 từng file.
- Chỉ xuất artifacts khi bật tùy chọn công khai; artifacts có thời hạn lưu 3 ngày.
- Kaggle CLI **không liệt kê toàn bộ version history** của một notebook từ lệnh `kernels list`; version cụ thể phải được cung cấp khi pull.
- Chưa có thao tác tự động ghi kết quả vào `Research_Report/KaggleNotebook`; sẽ triển khai sau khi nghiệm thu và quyết định phạm vi dữ liệu được phép công khai.

Tham khảo: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
