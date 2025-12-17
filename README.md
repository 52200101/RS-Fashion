# Dự Án CNTT - Hệ Thống Gợi Ý Sản Phẩm
Dự án nghiên cứu và triển khai các mô hình Graph-based Collaborative Filtering kết hợp với Content-Based Filtering cho hệ thống gợi ý sản phẩm, sử dụng PyTorch
    ## Tổng Quan

Dự án này triển khai ba mô hình học sâu phổ biến cho bài toán Collaborative Filtering

- **NGCF** - Mô hình học embedding người dùng và sản phẩm bằng cách lan truyền và kết hợp thông tin lân cận trên đồ thị User–Item thông qua các phép biến đổi phi tuyến.
- **LightGCN** -Phiên bản đơn giản hóa của NGCF, loại bỏ activation và transformation, chỉ giữ lại quá trình lan truyền embedding trên đồ thị để đạt hiệu quả cao và huấn luyện nhanh hơn.
- **ALS** - Phương pháp Collaborative Filtering truyền thống dựa trên phân rã ma trận User–Item, được sử dụng làm baseline để so sánh với các mô hình Graph-based.

## Cấu Trúc Dự Án
```
DACNTT/
├── src/
│   ├── evaluation/                 
|   │   └──metrics.py                # Đánh giá metrics cho các mô hình       
│   ├── models/
│   │   ├── NGCF.py                  # Mô hình NGCF (Graph-based CF)
│   │   ├── LightGCN.py              # Mô hình LightGCN
│   │   ├── ALS.py                   # Mô hình ALS
│   │   └── CBF.py                   # Content-Based Filtering và ensenble
│   ├── saved_models/                # Các Mô hình đã được train trên Kaggle và lưu về
│   │   ├── ngcf_full.pt             # Mô hình NGCF với toàn bộ dữ liệu
│   │   ├── ngcf_filter.pt           # Mô hình NGCF với bộ dữ liệu đã được lọc 
│   │   ├── lightgcn_full.pt         # Mô hình LightGCN với toàn bộ dữ liệu
│   │   ├── lightgcn_filter.pt       # Mô hình LightGCN với bộ dữ liệu đã được lọc
│   │   ├── als_full.pt              # Mô hình ALS với toàn bộ dữ liệu
│   │   └── als_filter.pt            # Mô hình ALS với bộ dữ liệu đã được lọc
│   └── utils/
│       ├── graph.py                 # Xây dựng đồ thị User–Item
│       └── load_model.py            # Load mô hình đã huấn luyện│
├── notebooks/
│   └── frs-dacntt.ipynb             # Notebook trên kaggle
│
├── data/
│   └── processed/                   # Dữ liệu đã xử lý
├── baocao.docx                      # File báo cáo của dự án
├── demo.py                          # Load các mô hình và đo metric của chúng
├── README.md                 
└── requirements.txt                 # Thư viện cần thiết
```

## Các Mô Hình

### 1. NGCF (Neural Graph Collaborative Filtering)
NGCF lan truyền embedding trên đồ thị User–Item, sau đó áp dụng biến đổi tuyến tính và phi tuyến để học biểu diễn người dùng và sản phẩm.
**Đặc điểm:**
-Mô hình hóa dữ liệu dưới dạng đồ thị hai phía (User–Item)
-Kết hợp self-connection và neighbor-connection
-Sử dụng activation function (ReLU/LeakyReLU)
-Biểu diễn giàu ngữ nghĩa nhưng chi phí tính toán cao hơn LightGCN
**Công thức:**
```
message = LeakyReLU(W_gc @ neighbor_emb) + LeakyReLU(W_bi @ (emb ⊙ neighbor_emb))
```
### 2. LightGCN (Light Graph Convolutional Network)
LightGCN đơn giản hóa NGCF bằng cách chỉ giữ lại bước lan truyền embedding trên đồ thị, không sử dụng biến đổi hay activation.
**Đặc điểm:**
-Không sử dụng activation function
-Không sử dụng feature transformation
-Giảm overfitting, tăng tốc độ huấn luyện
-Phù hợp với dữ liệu sparse
**Công thức:**
```
x^(k+1) = A_hat @ x^(k)
final_embedding = mean(x^(0), x^(1), ..., x^(K))
```
### 3. ALS (Alternating Least Squares)
ALS phân rã ma trận tương tác User–Item thành hai embedding thấp chiều và tối ưu luân phiên user và item.
**Đặc điểm:**
-Không sử dụng đồ thị
-Tối ưu luân phiên embedding user và item
-Dễ cài đặt, ổn định
-Được dùng làm baseline so sánh


### 4. Content-Based Filtering (CBF)
Content-Based Filtering gợi ý sản phẩm dựa trên **độ tương đồng nội dung** giữa các item mà user đã tương tác và các item còn lại.

**Đặc điểm:**
- Không phụ thuộc vào hành vi của user khác
- Sử dụng đặc trưng nội dung của item (ví dụ: category, tag, description)
- Giải quyết tốt bài toán cold-start cho item mới
- Được dùng để bổ sung cho Collaborative Filtering

---

### 5. Ensemble
Ensemble kết hợp kết quả từ nhiều mô hình khác nhau (NGCF, LightGCN, ALS, CBF) để tạo ra danh sách gợi ý cuối cùng.

**Đặc điểm:**
- Kết hợp điểm dự đoán từ nhiều mô hình
- Giảm bias của từng mô hình đơn lẻ
- Cải thiện độ ổn định và chất lượng gợi ý

**Cách thực hiện:**
- Chuẩn hóa score của từng mô hình
- Kết hợp bằng trung bình có trọng số hoặc cộng tuyến tính
- Sắp xếp lại item theo score tổng

---

## Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- CUDA (tùy chọn, để sử dụng GPU)

### Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Thư viện chính:
- `torch` -xây dựng và load model NGCF, LightGCN, ALS
- `numpy` - Xử lý mảng số
- `scikit-learn` - metric hỗ trợ

## Tối Ưu Siêu Tham Số

Các mô hình đã được tối ưu siêu tham số trước khi đánh giá, bao gồm:
- Embedding size
- Số tầng (layers)
- Learning rate
- Batch size
- Regularization

Cấu hình tốt nhất được chọn dựa trên kết quả trên tập validation.

---

## Kiến Trúc Kỹ Thuật

Hệ thống gợi ý được xây dựng theo kiến trúc offline, gồm các bước chính sau:

### 1. Xử lý dữ liệu
- Dữ liệu tương tác User–Item được tiền xử lý và lưu trong thư mục `data/processed`
- Tách dữ liệu thành tập train và test
- Lọc user và item có số tương tác quá ít (filter data)

---

### 2. Huấn luyện mô hình
- Xây dựng đồ thị User–Item bằng `graph.py`
- Huấn luyện các mô hình:
  - NGCF
  - LightGCN
  - ALS
- Các mô hình được train trên Kaggle và lưu lại dưới dạng `.pt` trong `saved_models`

---

### 3. Content-Based Filtering
- Trích xuất đặc trưng nội dung của item (category, tag, metadata)
- Tính độ tương đồng giữa các item bằng cosine similarity
- Sinh danh sách gợi ý dựa trên lịch sử tương tác của user

---

### 4. Ensemble
- Load kết quả dự đoán từ các mô hình:
  - Graph-based CF (NGCF, LightGCN)
  - ALS
  - Content-Based Filtering
- Chuẩn hóa score của từng mô hình
- Kết hợp score bằng trung bình hoặc trung bình có trọng số
- Sinh danh sách gợi ý cuối cùng cho mỗi user

---

### 5. Đánh giá mô hình
- So sánh danh sách gợi ý với tập test
- Tính các metric:
  - Precision@K
  - Recall@K
  - NDCG@K
  - HitRate@K
- Thực hiện đánh giá bằng `demo.py
