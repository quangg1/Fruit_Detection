# Đồ án: Phân loại giống trái cây (Fruit Variety Classification)

| Mục | Nội dung |
|-----|----------|
| **Môn học** | Digital Image Processing (DIP) |
| **Tên đề tài (tiếng Anh)** | *Fruit Variety Classification Using Image Processing Techniques* |
| **Ngôn ngữ / môi trường** | Python 3.x, Windows (khuyến nghị), có thể Linux/macOS |
| **Thư mục gốc mã nguồn** | `Digital_Image_Processing/` |

Tài liệu này mô tả **toàn bộ phạm vi**, **cấu trúc mã**, **luồng xử lý**, **cách cài đặt – huấn luyện – chạy**. Phần **[mục 9](#9-báo-cáo-quy-định-định-dạng-và-sườn-nội-dung-chi-tiết)** là **sườn báo cáo đầy đủ** (quy định in ấn + phần đầu báo cáo + Chương 1–4 + kết luận + **tài liệu tham khảo**) bám khung hướng dẫn đồ án — dùng để **chép dàn ý** sang Word/PDF.

### Mục lục tài liệu README này

| § | Nội dung |
|---|----------|
| [§1](#1-tóm-tắt-đồ-án-executive-summary) | Tóm tắt đồ án |
| [§2](#2-phạm-vi-dự-án-phiên-bản-hiện-tại) | Phạm vi dự án |
| [§3](#3-cấu-trúc-dự-án-và-vai-trò-từng-file) | Cấu trúc dự án & file |
| [§4](#4-dataset-định-dạng-và-quy-ước) | Dataset |
| [§5](#5-cài-đặt-môi-trường) | Cài đặt môi trường |
| [§6](#6-huấn-luyện-train) | Huấn luyện |
| [§7](#7-chạy-giao-diện-chính-streamlit) | Streamlit |
| [§8](#8-script-tùy-chọn-live-và-capture_samples) | Script tùy chọn (live, capture) |
| [§9](#9-báo-cáo-quy-định-định-dạng-và-sườn-nội-dung-chi-tiết) | **Báo cáo: quy định + sườn chi tiết** |
| [§10](#10-tài-liệu-liên-quan-trong-repo) | Tài liệu liên quan trong repo |
| [§11](#11-phiên-bản-và-cập-nhật) | Phiên bản & cập nhật |

---

## 1. Tóm tắt đồ án (Executive summary)

Đồ án xây dựng một **hệ thống phân loại ảnh trái cây theo giống** (ví dụ: Apple, Banana, Orange, …) dựa trên **xử lý ảnh số (DIP)** để trích **đặc trưng** từ ảnh, kết hợp **máy học có giám sát** (SVM) để **gán nhãn** và **ước lượng độ tin cậy**.

Điểm nhấn **DIP** trong pipeline:

- **Miền không gian:** lọc median, CLAHE trên kênh độ sáng trong không gian LAB.
- **Biến đổi hình học:** chọn vùng ROI (cắt vuông giữa khung, resize cố định).
- **Không gian màu:** HSV (histogram, có xử lý Hue để gom vùng đỏ hai đầu thang OpenCV), LAB (thống kê kênh a, b).
- **Kết cấu:** LBP (Local Binary Pattern) với histogram cố định số bin.
- **Phân loại:** SVM kernel RBF, xác suất lớp qua `predict_proba`.

**Giao diện chính** là ứng dụng **Streamlit** (`streamlit_fruit_app.py`): người dùng **tải ảnh**, xem **từng bước tiền xử lý và kênh màu**, nhận **nhãn dự đoán** và **biểu đồ xác suất** theo từng loại trái.

**Webcam / video real-time** không nằm trong phạm vi bắt buộc của đồ án; có thể dùng thêm script `live.py` hoặc `capture_samples.py` nếu muốn thử nhanh với camera.

---

## 2. Phạm vi dự án (phiên bản hiện tại)

### 2.1. Những gì đã được triển khai và coi là “đủ” cho đồ án

| Thành phần | Chi tiết cụ thể | File / vị trí |
|------------|-----------------|----------------|
| **Tiền xử lý DIP** | Lọc **median** (giảm nhiễu điểm đơn lẻ, giữ biên tốt hơn Gaussian cho chi tiết vỏ). **CLAHE** chỉ trên kênh **L** của **LAB** (cân tương phản cục bộ, ít phụ thuộc chỗ sáng/tối toàn khung). | `fruit_project/features.py` — hàm `dip_preprocess`, lớp `DIPOptions` |
| **ROI** | Cắt **hình vuông lớn nhất ở giữa** khung hình; **resize** về kích thước cố định (mặc định **128×128**, có thể chỉnh trong UI Streamlit hoặc `DIPOptions` — **phải trùng lúc train và lúc dự đoán**). | `preprocess_for_fruit` |
| **Đặc trưng màu** | Histogram **H, S, V** trong HSV; với **H** dùng **ánh xạ gộp** hai vùng đỏ ở hai đầu thang Hue của OpenCV (0–179) trước khi histogram; thêm **mean / std** kênh **a** và **b** của LAB. | `extract_hsv_histogram`, `extract_lab_color_stats` |
| **Đặc trưng kết cấu** | **LBP** uniform, P=8, R=1; **histogram 11 bin** cố định (0–10), chuẩn hóa L2 từng phần; **vector cuối** (HSV + Lab stats + LBP) được **chuẩn hóa L2 toàn cục** một lần. | `extract_lbp_hist`, `extract_features` |
| **Huấn luyện** | Đọc ảnh **đệ quy** trong `data/fruits/<tên_lớp>/` (mọi `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). **Xáo trộn** và tùy chọn **giới hạn số ảnh mỗi lớp** (`--max-per-class`) để train nhanh hơn trên dataset lớn. **SVM** `kernel='rbf'`, `probability=True`. Lưu **model + LabelEncoder** bằng `joblib`; lưu **metadata** (classes, số mẫu, …) ra JSON. | `fruit_project/train.py` |
| **Giao diện** | **Streamlit**: upload ảnh, bật/tắt DIP, chỉnh tham số median/CLAHE/ROI, hiển thị ảnh từng bước, metric nhãn + độ tin cậy, biểu đồ xác suất. | `streamlit_fruit_app.py` |

### 2.2. Phần tùy chọn (không bắt buộc)

| Mục đích | Lệnh | Ghi chú |
|----------|------|---------|
| Nhận dạng qua **cửa sổ OpenCV** (webcam laptop hoặc URL IP camera) | `python -m fruit_project.live` | Biến môi trường `FRUIT_CAMERA`: `0` = camera đầu tiên; hoặc `http://IP:8080/video` (IP Webcam Android). |
| Thu thêm **ảnh mẫu** vào một lớp | `python -m fruit_project.capture_samples <tên_lớp> [--camera ...]` | **Space** lưu ảnh, **q** thoát. |

---

## 3. Cấu trúc dự án và vai trò từng file

### 3.1. Sơ đồ thư mục (chi tiết)

```
Digital_Image_Processing/
├── fruit_project/
│   ├── __init__.py
│   ├── config.py           # Đường dẫn DATA_DIR, MODEL_PATH, META_PATH; FRUIT_CAMERA; parse_camera_source()
│   ├── features.py         # Toàn bộ DIP + trích đặc trưng + pipeline_visualization (ảnh từng bước cho UI)
│   ├── train.py            # load_dataset(), train SVM, ghi weights + fruit_meta.json
│   ├── live.py             # Vòng lặp OpenCV + predict_proba (tùy chọn)
│   └── capture_samples.py  # Lưu ảnh từ camera vào data/fruits/<lớp>/ (tùy chọn)
├── data/
│   └── fruits/
│       ├── <Lớp_1>/        # Ví dụ Apple, Banana — chứa ảnh hoặc thư mục con chứa ảnh
│       ├── <Lớp_2>/
│       └── README.txt
├── weights/
│   ├── fruit_svm.joblib   # dict: model (SVC), label_encoder (LabelEncoder)
│   └── fruit_meta.json    # classes, n_samples, max_per_class, feature_dim, ...
├── streamlit_fruit_app.py # Giao diện web chính
├── requirements.txt       # opencv-python, numpy, scikit-learn, scikit-image, streamlit, pandas, joblib, ...
├── run_fruit_ui.bat       # Windows: chạy Streamlit nhanh
└── README_FRUIT_PROJECT.md
```

### 3.2. Luồng dữ liệu từ ảnh đến nhãn (một dòng)

1. Ảnh **BGR** (`uint8`), từ file hoặc camera.  
2. **`dip_preprocess`**: median → CLAHE trên L (LAB).  
3. **`preprocess_for_fruit`**: cắt giữa, resize → ROI cố định.  
4. **`extract_hsv_histogram`** + **`extract_lab_color_stats`** + **`extract_lbp_hist`**.  
5. **Nối vector** → **chuẩn hóa L2** toàn vector.  
6. **`clf.predict_proba`** → argmax → **nhãn** + **xác suất**.

---

## 4. Dataset: định dạng và quy ước

### 4.1. Cấu trúc thư mục

- Mỗi **lớp** (một giống trái) = **một thư mục con** trực tiếp dưới `data/fruits/`.  
- Tên thư mục = **nhãn** (ví dụ `Apple`, `Orange`) — sẽ khớp với `LabelEncoder` sau khi train.  
- Ảnh có thể nằm **trực tiếp** trong thư mục lớp hoặc **trong thư mục con** (train đọc **đệ quy** `rglob`).  
- Định dạng hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

### 4.2. Số lượng mẫu

- Tối thiểu để train chạy: **ít nhất 2 lớp** và **ít nhất 4 ảnh** tổng (theo logic trong `train.py`).  
- Thực tế nên **vài chục ảnh trở lên mỗi lớp**; với dataset lớn (hàng nghìn ảnh/lớp) nên dùng  
  `python -m fruit_project.train --max-per-class 1500`  
  (hoặc 800–2000) để **thời gian train SVM** và **bộ nhớ** chấp nhận được.

### 4.3. Lưu ý “miền ảnh” (domain)

Model học trên **phân phối** của ảnh trong `data/fruits/`. Ảnh **minh họa / clipart / nền đen phẳng** có histogram và LBP **rất khác** ảnh chụp thật → SVM có thể **dự đoán sai** dù độ tin cậy cao. Khi demo báo cáo, nên ưu tiên **ảnh chụp thật** cùng kiểu với dataset.

---

## 5. Cài đặt môi trường

### 5.1. Điều kiện

- **Python** 3.10+ (khuyến nghị; đã kiểm tra với 3.8+).  
- **pip** để cài đặt gói.

### 5.2. Cài đặt

Trong thư mục gốc project:

```powershell
cd d:\Digital_Image_Processing
python -m pip install -r requirements.txt
```

### 5.3. Các gói chính trong `requirements.txt`

| Gói | Vai trò |
|-----|---------|
| `opencv-python` | Đọc ảnh, xử lý pixel, không gian màu, median, CLAHE, histogram |
| `numpy` | Mảng số, vector đặc trưng |
| `scikit-learn` | SVM, LabelEncoder |
| `scikit-image` | LBP |
| `scipy` | (phụ thuộc của scikit-image / sklearn) |
| `matplotlib` | (tuỳ dùng cho vẽ thêm; có thể bổ sung sau) |
| `Pillow` | (tuỳ thư viện) |
| `joblib` | Lưu/tải model |
| `streamlit` | Giao diện web |
| `pandas` | Biểu đồ trong Streamlit |

---

## 6. Huấn luyện (train)

### 6.1. Lệnh cơ bản

```powershell
cd d:\Digital_Image_Processing
python -m fruit_project.train
```

Lệnh này đọc **toàn bộ** ảnh hợp lệ trong mỗi lớp (có thể **rất lâu** nếu dataset cực lớn).

### 6.2. Giới hạn số ảnh mỗi lớp (khuyến nghị)

```powershell
python -m fruit_project.train --max-per-class 1500
```

- **Tác dụng:** sau khi **xáo trộn** danh sách ảnh trong mỗi lớp, chỉ lấy **tối đa N ảnh** để train → **nhanh hơn**, **ít RAM**.  
- **Tham số:** `--seed` (mặc định 42) để tái lập kết quả khi cần.

### 6.3. Đầu ra

| File | Nội dung |
|------|----------|
| `weights/fruit_svm.joblib` | `{"model": SVC, "label_encoder": LabelEncoder}` |
| `weights/fruit_meta.json` | JSON: `classes`, `n_samples`, `feature_dim`, `max_per_class`, … |

**Sau khi sửa** `features.py` (thay đổi chiều vector đặc trưng), **bắt buộc train lại** trước khi dùng Streamlit / live, nếu không vector test không khớp kích thước / phân phối đã học.

---

## 7. Chạy giao diện chính (Streamlit)

### 7.1. Lệnh

```powershell
cd d:\Digital_Image_Processing
streamlit run streamlit_fruit_app.py
```

Trình duyệt mở (thường `http://localhost:8501`).

### 7.2. Chức năng trên giao diện

- **Sidebar:** hiển thị pipeline bắt buộc (Spatial enhancement -> Feature extraction -> Pattern recognition), chỉnh median ksize, CLAHE clip, **kích thước ROI** (phải **khớp** với lúc train nếu đã đổi mặc định); trạng thái model (đã có file hay chưa).  
- **Vùng chính:** upload ảnh → hiển thị lần lượt: ảnh gốc → sau median+CLAHE → ROI → kênh H, S, V (minh họa).  
- Nếu có `weights/fruit_svm.joblib`: hiển thị **nhãn dự đoán**, **độ tin cậy**, **biểu đồ cột** xác suất theo từng lớp.  
- Có **expanders** giải thích DIP và **tại sao đôi khi nhầm lớp** (lệch miền, Hue đỏ trong OpenCV).

### 7.3. Windows nhanh

Double-click `run_fruit_ui.bat` (nếu có) hoặc gõ lệnh tương đương trong PowerShell.

---

## 8. Script tùy chọn: `live` và `capture_samples`

### 8.1. `fruit_project.live`

- Mở **VideoCapture** (camera số hoặc URL HTTP từ IP Webcam).  
- Mỗi khung: `extract_features` → `predict_proba` → vẽ text lên khung.  
- **Thoát:** phím `q` trong cửa sổ OpenCV.

```powershell
python -m fruit_project.live
```

Camera URL:

```powershell
$env:FRUIT_CAMERA="http://192.168.1.10:8080/video"
python -m fruit_project.live
```

### 8.2. `fruit_project.capture_samples`

Thu ảnh vào `data/fruits/<tên_lớp>/` (phím Space lưu, q thoát).

```powershell
python -m fruit_project.capture_samples Apple --camera "http://IP:8080/video"
```

---

## 9. Báo cáo: quy định định dạng và sườn nội dung chi tiết

Phần này thay thế hoàn toàn cho “gợi ý ngắn”: là **khung viết báo cáo** (TLCN / ĐATN / tiểu luận môn) bám **quy định chung** và **gợi ý cấu trúc** như tài liệu hướng dẫn của khoa. Khi chuyển sang Word, áp dụng **đúng thông số in**; nội dung dưới đây là **đề mục + gợi ý đoạn** (có thể copy rồi chỉnh văn phong học thuật).

### 9.1. Quy định chung về định dạng văn bản

| Yếu tố | Quy định | Ghi chú khi soạn |
|--------|----------|------------------|
| Khổ giấy | **A4** | — |
| Cỡ chữ | **13** | Thường dùng Times New Roman hoặc theo quy định khoa |
| Lề | **Trái 3 cm**; **phải, trên, dưới 2 cm** | Kiểm tra “Page Setup” trước khi nộp |
| Giãn dòng | **1.3** dòng | Không dùng single spacing quá sát |
| Bắt đầu chương | **Mỗi chương lớn** (Chương 1, 2, …) **bắt đầu trang mới** | Phần “Mở đầu” có thể gồm nhiều mục nhỏ trên cùng nhóm trang tùy quy định giảng viên |

---

### 9.2. Phần đầu báo cáo (Front matter)

#### Trang bìa (Cover page)

Nội dung **bắt buộc** (điền theo mẫu trường):

- Tên **trường đại học**, **khoa** / bộ môn.  
- Loại đồ án: **TLCN** / **ĐATN** / **bài tiểu luận** (theo đúng tên gọi môn học yêu cầu).  
- **Tên đề tài** (tiếng Việt và/hoặc tiếng Anh): ví dụ *Phân loại giống trái cây dựa trên kỹ thuật xử lý ảnh và nhận dạng mẫu* / *Fruit Variety Classification Using Image Processing Techniques*.  
- **Giảng viên hướng dẫn** (họ tên).  
- **Mã số sinh viên**, **họ tên sinh viên** (nhóm thì ghi đủ thành viên nếu có).  
- **Năm** / học kỳ.

#### Mục lục (Table of Contents)

- Liệt kê **tất cả** mục đến cấp **3** (nếu báo cáo có); **số trang** đối chiếu — cập nhật sau cùng.

#### Danh mục từ viết tắt / cụm từ (List of Abbreviations)

Gợi ý các mục cần giải thích lần đầu trong báo cáo:

| Ký hiệu / từ viết tắt | Diễn giải đầy đủ |
|------------------------|------------------|
| DIP | Digital Image Processing — Xử lý ảnh số |
| ROI | Region of Interest — Vùng quan tâm |
| HSV | Hue – Saturation – Value — Không gian màu |
| LAB / L*a*b* | Không gian màu CIE L*a*b* (OpenCV: LAB) |
| LBP | Local Binary Pattern |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |
| SVM | Support Vector Machine |
| GUI / UI | Giao diện người dùng |
| API | Application Programming Interface (nếu có đề cập) |

#### Danh sách hình vẽ (List of Figures)

Gợi ý hình nên có trong báo cáo (đánh số Hình 1.1, 2.1, …):

1. **Sơ đồ khối** toàn pipeline: từ ảnh đầu vào → tiền xử lý → ROI → trích đặc trưng → SVM → nhãn.  
2. Minh họa **từng bước DIP** (ảnh gốc, sau median+CLAHE, ROI, kênh H/S/V) — có thể chụp từ Streamlit.  
3. **Giao diện** Streamlit: màn hình upload + kết quả + biểu đồ xác suất.  
4. (Tùy chọn) **Confusion matrix** nếu có đánh giá trên tập kiểm tra.  
5. (Tùy chọn) Ví dụ **nhầm lớp** (ảnh minh họa vs ảnh chụp) để minh họa lệch miền.

#### Danh sách bảng biểu (List of Tables)

Gợi ý:

- Bảng **thống kê dataset** (tên lớp, số ảnh).  
- Bảng **tham số huấn luyện** (`max_per_class`, `seed`, kernel SVM, `C`, `gamma`).  
- Bảng **kết quả** (accuracy từng lớp / toàn cục) nếu có đo.  
- Bảng **cấu hình phần cứng** phần thực nghiệm.

---

### 9.3. Mở đầu — mô tả về project (Introduction)

*Phần này tương ứng mục “Mở đầu” trong khung hướng dẫn; có thể đặt tên chương “Chương 1. Mở đầu” hoặc “Phần mở đầu” tùy mẫu khoa.*

#### Đặt vấn đề (Problem statement)

- **Bối cảnh:** Trong giáo dục và ứng dụng nhận dạng đơn giản, cần **phân loại đối tượng trong ảnh** mà không luôn phải dùng mạng nơ-ron lớn; **DIP kết hợp đặc trưng thủ công + máy học cổ điển** vẫn là nền tảng quan trọng.  
- **Vấn đề cụ thể:** Phân loại **giống trái cây** từ ảnh một hoặc nhiều trái nằm trong khung, đầu ra là **nhãn lớp** (Apple, Banana, …) và **độ tin cậy**.  
- **Thách thức:** Ảnh chụp thực tế có **nhiễu, ánh sáng không đều, nền phức tạp**; cần **tiền xử lý** và **vector đặc trưng ổn định**; **lệch miền** giữa ảnh train và ảnh test (ví dụ ảnh vẽ) có thể làm giảm độ chính xác cảm quan.

#### Mục tiêu và yêu cầu (Objectives and requirements)

- **Mục tiêu chính:** Xây dựng pipeline **DIP → đặc trưng → SVM**, huấn luyện trên `data/fruits/`, lưu model, triển khai **giao diện web** để người dùng tải ảnh và nhận **nhãn + xác suất**.  
- **Yêu cầu chức năng:**  
  - Tiền xử lý: median, CLAHE (LAB); ROI cố định.  
  - Trích đặc trưng: histogram HSV (Hue đã xử lý đỏ), S, V; thống kê a,b (LAB); LBP.  
  - Phân loại: SVM, xác suất lớp.  
  - Giao diện: Streamlit, hiển thị pipeline trực quan.  
- **Yêu cầu phi chức năng (gợi ý):** Chạy được trên PC thông thường; có thể giới hạn số mẫu train (`--max-per-class`) khi dataset lớn.

#### Phạm vi và đối tượng (Scope and subjects)

- **Phạm vi:**  
  - **Đối tượng:** ảnh **tĩnh** (file); **không** bắt buộc hệ thống **video real-time** hay webcam trên trình duyệt trong phạm vi đồ án hiện tại.  
  - **Số lớp:** Theo thư mục con trong `data/fruits/` (ví dụ 6 lớp).  
  - **Không** mở rộng sang phân đoạn instance-level phức tạp (nhiều trái chồng lấn) trừ khi báo cáo mở rộng thêm.  
- **Đối tượng sử dụng:** Sinh viên / giảng viên chấm đồ án; minh họa môn DIP.

#### Cấu trúc báo cáo (Overview of report structure)

Đoạn ngắn (nửa trang): “Báo cáo gồm X chương. Chương 2 trình bày cơ sở lý thuyết… Chương 3 trình bày thiết kế… Chương 4 trình bày thực nghiệm… Kết luận tóm tắt kết quả và hướng phát triển.”

---

### 9.4. Chương 2. Cơ sở lý thuyết dùng để thực hiện project

#### Nghiên cứu các hướng giải pháp liên quan (Related work / solutions)

- **Hướng 1 — Đặc trưng thủ công + bộ phân loại cổ điển (SVM, k-NN, Random Forest):**  
  - Ưu: Giải thích được từng bước DIP, phù hợp đồ án môn DIP.  
  - Nhược: Cần thiết kế đặc trưng tốt; hiệu năng phụ thuộc dataset.  
- **Hướng 2 — Học sâu (CNN):**  
  - Ưu: Độ chính xác cao trên dataset lớn.  
  - Nhược: Cần dữ liệu và tài nguyên; ít “mở hộp đen” hơn cho báo cáo DIP thuần.  
- **Kết luận lựa chọn:** Đồ án chọn **hướng 1** để **đi sát giáo trình DIP** (median, CLAHE, histogram, LBP).

#### Môi trường lập trình, công cụ và thư viện (Tools and environment)

- **Ngôn ngữ:** Python 3.x.  
- **Thư viện:**  
  - **OpenCV:** đọc ảnh, không gian màu, median, CLAHE, histogram.  
  - **NumPy:** vector hóa.  
  - **scikit-image:** LBP.  
  - **scikit-learn:** SVM, `LabelEncoder`, `joblib`.  
  - **Streamlit:** giao diện web.  
  - **pandas:** hỗ trợ biểu đồ trong Streamlit.  
- **Môi trường:** Windows/Linux; **IDE** (VS Code, PyCharm, …) ghi rõ trong báo cáo.

#### Các phương pháp và kỹ thuật DIP / nhận dạng (chi tiết — mỗi mục 1–2 trang)

1. **Lọc median:** Nguyên lý cửa sổ lân cận, trung vị; ứng dụng giảm nhiễu muối tiêu; so sánh ngắn với Gaussian.  
2. **CLAHE trên kênh L (LAB):** Bản chất histogram cục bộ có giới hạn contrast; vì sao áp dụng trên **L** chứ không phải toàn RGB.  
3. **ROI — cắt giữa + resize:** Chuẩn hóa không gian và kích thước; giả định trái ở giữa khung.  
4. **Không gian HSV và histogram:** Ý nghĩa H, S, V; vấn đề **Hue đỏ** hai đầu thang OpenCV và cách **gộp** trong đồ án.  
5. **Không gian LAB và kênh a, b:** Trục đỏ–xanh, vàng–xanh; mean/std trên ROI.  
6. **LBP:** Ý tưởng so sánh pixel với lân cận; uniform pattern; histogram cố định số bin.  
7. **SVM:** Siêu phẳng, kernel RBF, ý nghĩa tham số `C`, `gamma`; `probability=True` để có xác suất.  
8. **Chuẩn hóa vector (L2):** Ổn định thang đo trước SVM.

*(Mỗi mục nên có **hình minh họa** hoặc **công thức** trích từ giáo trình Gonzalez & Woods, hoặc tài liệu OpenCV.)*

---

### 9.5. Chương 3. Phân tích và thiết kế giải pháp

#### Sơ đồ khối và ý tưởng thuật toán (Flowcharts / algorithm ideas)

- **Sơ đồ tổng:** Khối “Ảnh đầu vào” → “Tiền xử lý DIP” → “ROI” → “Trích đặc trưng” → “Vector” → “SVM” → “Nhãn + xác suất”.  
- **Sơ đồ con (tùy chọn):** Chi tiết `extract_features` (nhánh HSV, Lab stats, LBP).  
- **Giả mã (pseudo-code)** cho `train.py`: duyệt lớp → đọc ảnh → `extract_features` → stack `X`, `y` → `fit` SVM → `joblib.dump`.

#### Phân tích và mô tả thuật toán chính (Main algorithms)

- **Thuật toán 1 — Tiền xử lý `dip_preprocess`:** Đầu vào/ra, tham số `DIPOptions`.  
- **Thuật toán 2 — `extract_features`:** Thứ tự gọi hàm, chiều vector, chuẩn hóa L2.  
- **Thuật toán 3 — Huấn luyện SVM:** Kernel, xác suất, lưu `LabelEncoder` để map chỉ số ↔ tên lớp.  
- **Thuật toán 4 — Dự đoán trên Streamlit:** `predict_proba`, argmax, hiển thị.

#### Thiết kế phần mềm (Software design)

- **Module `features.py`:** DIP + đặc trưng + `pipeline_visualization`.  
- **Module `train.py`:** Đọc dataset đệ quy, train, metadata JSON.  
- **`streamlit_fruit_app.py`:** Tương tác người dùng, đồng bộ tham số với `DIPOptions`.  
- **`config.py`:** Đường dẫn tập trung.

---

### 9.6. Chương 4. Thực nghiệm, đánh giá và phân tích kết quả

#### Quy trình thực nghiệm (Experimental process)

- Mô tả **chuẩn bị dữ liệu:** cấu trúc `data/fruits/`, số lớp, ước lượng số ảnh; có hay không dùng `--max-per-class`.  
- **Cấu hình phần cứng:** CPU, RAM, OS, (GPU nếu không dùng thì ghi rõ “không sử dụng GPU”).  
- **Siêu tham số (hyperparameters):** `max_per_class`, `seed`, tham số SVM trong code (`C=10`, `gamma='scale'`), tham số median/CLAHE mặc định.

#### Kết quả thực nghiệm (Experimental results)

- **Định lượng (nếu thực hiện):** Chia train/validation (có thể script bổ sung), báo **accuracy**, **confusion matrix**, **precision/recall** từng lớp.  
- **Định tính:** Ảnh chụp màn hình Streamlit; phân tích **trường hợp đúng** và **trường hợp sai** (clipart, nền đen, lệch miền).  
- **Thời gian:** Thời gian train ước lượng (phút) với cấu hình đã ghi.

#### Giao diện phần mềm và hướng dẫn chạy (Software interface & user guide)

- **Mô tả giao diện:** Sidebar, vùng upload, các ảnh pipeline, biểu đồ xác suất.  
- **Hướng dẫn từng bước cho người dùng cuối:** Cài `requirements.txt` → `train` (nếu chưa có model) → `streamlit run streamlit_fruit_app.py` → upload ảnh → đọc kết quả.  
- **Script tùy chọn:** `live.py`, `capture_samples.py` — mục đích, không bắt buộc trong phạm vi tối thiểu.

---

### 9.7. Kết luận (Conclusion)

#### Đánh giá kết quả đạt được (Evaluation of achieved results)

- Đã hoàn thành: pipeline DIP + đặc trưng + SVM + lưu model + giao diện Streamlit.  
- Điểm mạnh: Giải thích được từng bước; trực quan hóa pipeline.  
- Điểm hạn chế: Phụ thuộc giả định trái ở giữa khung; nhạy với lệch miền ảnh; SVM trên vector cố định có thể cần tối ưu tham số hoặc thêm dữ liệu.

#### Hướng phát triển (Future directions)

- Thu thêm dữ liệu đa dạng; tách tập validation; đo chỉ số hệ thống.  
- Thử **data augmentation** nhẹ (xoay, đổi sáng) trong phạm vi DIP.  
- (Tùy chọn) Mở rộng **real-time** qua camera — **không** bắt buộc với phạm vi đồ án hiện tại đã mô tả ở README dự án.

---

### 9.8. Tài liệu tham khảo (References)

Quy định: **Liệt kê** tài liệu đã trích dẫn trong báo cáo; **trích dẫn trong văn** theo chuẩn khoa (VD: [1], [2] hoặc Harvard).

**Gợi ý nguồn trích dẫn:**

1. Gonzalez, R. C., & Woods, R. E. — *Digital Image Processing* (hoặc ấn bản/phiên bản giáo trình môn học).  
2. Tài liệu **OpenCV**: [https://docs.opencv.org/](https://docs.opencv.org/) — không gian màu, `calcHist`, CLAHE, median.  
3. **scikit-learn** — User Guide: SVM, `SVC`.  
4. **scikit-image** — Local Binary Pattern.  
5. **Streamlit** — Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)  
6. Bài báo / khóa học về LBP (Ojala et al.) nếu cần trích công trình gốc.

*(Trong file báo cáo Word, định dạng tham khảo theo **một** chuẩn thống nhất — Ví dụ IEEE hoặc APA — theo yêu cầu khoa.)*

---

## 10. Tài liệu liên quan trong repo

| File | Mục đích |
|------|----------|
| `README_FRUIT.md` | Hướng dẫn ngắn: train, Streamlit, ghi chú tùy chọn IP Webcam |
| `README.md` | Ghi chú đăng ký đề tài nhóm (nếu lớp dùng chung repo) |

---

## 11. Phiên bản và cập nhật

- README này phản ánh **phiên bản mã** tại thời điểm soạn: module `fruit_project` với `features.py` (median, CLAHE, HSV gộp đỏ, Lab stats, LBP, L2), `train.py`, `streamlit_fruit_app.py`.  
- Khi thêm tính năng (ví dụ đánh giá confusion matrix trên tập riêng), nên **cập nhật** mục Chương 4 và bảng “Đầu ra” trong README này.

---

*Tài liệu này có thể copy vào báo cáo sau khi chỉnh lại văn phong học thuật và đánh số hình/bảng theo quy định khoa.*
