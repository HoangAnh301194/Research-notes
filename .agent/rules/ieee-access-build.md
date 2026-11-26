---
trigger: glob
globs:
  - Research_Report/IEEE_Access/**
---

# Quy tắc biên dịch bài báo IEEE Access

- **Thư mục làm việc:** `Research_Report/IEEE_Access` (file chính: `main.tex`).
- **Template & Font:** Sử dụng class `ieeeaccess.cls` và bộ font trong `Research_Report/IEEE_Access/template/`. Không thay thế bằng `IEEEtran`.
- **Cách build PDF (MiKTeX trên Windows):** Luôn chạy 2 lượt với cờ `-include-directory=template` để MiKTeX tìm đủ `.cls`, `.sty`, `.tfm`, `.map` và `.pfb` (không chỉ set `TEXINPUTS` vì sẽ thiếu font):
  ```powershell
  cd D:\PTIT\NCKH\Research-notes\Research_Report\IEEE_Access
  $exe = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
  & $exe -include-directory=template -interaction=nonstopmode main.tex
  & $exe -include-directory=template -interaction=nonstopmode main.tex
  ```
- **Hình ảnh:** Lưu các hình đã xuất (`Fig1.pdf`, `Fig2.pdf`, `Fig3.pdf`, ...) trong `Research_Report/IEEE_Access/Figures/`.
