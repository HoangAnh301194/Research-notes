# Mamba : Linear - Time Sequence Modeling with Selective State Space 

- Báo cáo tóm tắt nội dung tìm hiểu bài báo : Mamba : Mô hình hóa chuỗi với thời gian tuyến tính bằng mô hình không gian có chọn lọc .

## Abstract
- Đa số các mô hình học sâu hiện nay đều ứng dụng kiến trúc Transformer và các mô hình based on attention module .
- Để giải quyết vấn đề kém hiệu quả trong tính toán với chuỗi thông tin dài thì người ta đã đưa ra các kiến trúc thấp hơn bậc 2 (abquadratic-time architectures) như **linear attention**, **gated convolution**, **recurrent models** và **mô hình không gian trạng thái có cấu trúc (structured state space models-SSMs)**. Tuy nhiên những kiến trúc này khôgn hiệu quả tốt bằng attention trong các bài toán phân tích dữ liệu quan trọng nhưu ngôn ngữ. 

- Tác giải đưa ra phương pháp để giải quyết vấn đề trên như sau : 
    - Cho phép các tham số SSM trở thành các hàm phụ thuộc dữ liệu dầu vào để khắc phục điểm yếu của SSM đối với các dạng dữ liệu rời rạc (discrete modalities)
    - Thiết kế một thuật toán song song có xét tới đặc tính của phần cứng để thực hiện mô hình ở chế độ hồi quy (recurrent mode )

- Kiến trúc mô hình **Mamba** được thiết kế bởi các selective SSMs tích hợp với các mạng neural network end to end một cách đơn giản, không cần attention và các khối MLP blocks. 
- Kết quả thu được : 
    - Tốc độ **inference time** gấm **5 lần** Transformer và tỷ lệ tuyến tính theo chiều dài chuỗi 
    - Hiệu năng mô hình cải tiện dữ liệu thực lên tới các chuỗi có độ dài hàng triệu phần tử. 

- Mô hình **Mamba - 3B** vượt trội hơn **transformer** có cùng kích thước và đạt hiệu năng tương đường vs các Transformer có kích thước gấp đôi 

## Introduction 
