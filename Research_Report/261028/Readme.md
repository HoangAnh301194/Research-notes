# Báo cáo nghiên cứu ngày 28/10/2026

## A. Công việc đã làm. 
- Đóng gói dataset EgoEv Gesture lên Kaggle , tiến hành train 
- Xuất keyframe của 2 model Videomamba train trên 2 tập dataset IPN Hand và CTP Gesture

### 1. Đóng gói dataset EgoEv Gesture 
- cấu trúc dataset 

> Hạn chế : dữ liệu subj 5 và 10 là giống hệt nhau và có 1 file trùng giữa subj 4 và 8 
> Khi tiến hành chia thì loại bỏ trùng 

- Split  dataset 
    - train
    - val 
    - test 

- Cấu hình khi train 
- Lưu lại checkpoit , mỗi phiên session train 8 epoch 
- Tuy nhiên hiện tại đã train tới 16 epoch nhưng Acc vẫn tăng thêm , chỉ dừng ở mức 67% nên em tạm thời dừng train 


### 2. Xuất ra keyframe correct and error predection để quan sát 

#### 2.1 IPN Hand 


### 2.2 CTP Gesture 


## B. Khó khăn
- không
## C. Công việc tiếp theo
- 