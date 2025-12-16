# Dự Án CNTT - Hệ Thống Gợi Ý Sản Phẩm
Dự án nghiên cứu và triển khai các mô hình Graph-based Collaborative Filtering kết hợp với Content-Based Filtering cho hệ thống gợi ý sản phẩm, sử dụng PyTorch
    ## Tổng Quan

Dự án này triển khai ba mô hình học sâu phổ biến cho bài toán Collaborative Filtering

- **NGCF** - Mô hình học embedding người dùng và sản phẩm bằng cách lan truyền và kết hợp thông tin láng giềng trên đồ thị User–Item thông qua các phép biến đổi phi tuyến.
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
**Công thức:**

