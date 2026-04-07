# Fruit Variety Classification — hướng dẫn nhanh

**Phạm vi chính:** huấn luyện trên `data/fruits/` + giao diện **Streamlit** (tải ảnh phân loại). **Webcam / IP camera** chỉ là **tùy chọn** khi thu mẫu hoặc chạy `live`.

Chi tiết và cấu trúc báo cáo: **`README_FRUIT_PROJECT.md`**.

---

## Điện thoại làm webcam (Android — IP Webcam) — tùy chọn

1. Cài **IP Webcam** (hoặc app tương tự) trên điện thoại.
2. Mở app → bật server → ghi **địa chỉ** dạng `http://192.168.x.x:8080`.
3. Điện thoại và **máy tính cùng Wi‑Fi**.
4. Trên PC, luồng video thường là:
   ```text
   http://<IP_điện_thoại>:8080/video
   ```
5. Thử mở URL đó trên trình duyệt PC — thấy hình là được.

**Gán biến môi trường (PowerShell):**
```powershell
$env:FRUIT_CAMERA="http://192.168.1.50:8080/video"
```

**Hoặc truyền thẳng khi chụp mẫu:**
```powershell
python -m fruit_project.capture_samples apple --camera "http://192.168.1.50:8080/video"
```

*(iPhone: thường dùng app kiểu EpocCam / tương tự — xem URL luồng trong app.)*

## Thu dữ liệu

```powershell
cd d:\Digital_Image_Processing
pip install -r requirements.txt
```

Tạo thư mục `data/fruits/ten_loai/` và chụp (đặt trái **giữa khung**):

```powershell
python -m fruit_project.capture_samples apple --camera "http://IP:8080/video"
```

Lặp cho `banana`, `orange`, … (phím **Space** lưu ảnh, **q** thoát).

## Train

```powershell
python -m fruit_project.train
```

Dataset lớn (hàng chục nghìn ảnh): nên **giới hạn mỗi lớp** để SVM train nhanh, ví dụ:

```powershell
python -m fruit_project.train --max-per-class 1500
```

Model lưu tại `weights/fruit_svm.joblib`.

Ảnh có thể nằm **trong thư mục con** (ví dụ `Apple/Apple A/...`) — script đọc **đệ quy** toàn bộ file ảnh.

## Giao diện Streamlit (khuyến nghị)

```powershell
streamlit run streamlit_fruit_app.py
```

Hoặc `run_fruit_ui.bat` — xem pipeline DIP từng bước + biểu đồ xác suất.

## Chạy nhận dạng trực tiếp (OpenCV cửa sổ)

```powershell
$env:FRUIT_CAMERA="http://IP:8080/video"
python -m fruit_project.live
```

Nếu dùng webcam laptop: `set FRUIT_CAMERA=0` hoặc xóa biến môi trường.

**Lưu ý:** Đã bổ sung **median + CLAHE (LAB)** trong `extract_features`. Nên **train lại** model sau khi cập nhật code: `python -m fruit_project.train --max-per-class 1500`.

## Kỹ thuật (báo cáo DIP)

- Tiền xử lý: cắt vuông giữa, resize.
- Đặc trưng: histogram **HSV** + **LBP** (kết cấu).
- Phân loại: **SVM** (RBF), xác suất `predict_proba`.
