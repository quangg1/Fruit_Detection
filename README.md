# Digital Image Processing

**Đồ án phân loại trái cây (Fruit classification):** xem **`README_FRUIT_PROJECT.md`** — mô tả đầy đủ, cấu trúc báo cáo và hướng dẫn chạy (Streamlit + train; **không** yêu cầu webcam).

---

## Các đề tài đã đăng ký (theo ảnh bảng nhóm)

Chỉ ghi nhận **các đề có trong ảnh** bạn cung cấp (DIPR430685E — Nhóm 01FIE). Không bổ sung đề từ nguồn khác.

| Nhóm | Tên đề tài |
|------|------------|
| 1 | Hand Gesture-Controlled Interactive Team Portfolio |
| 2 | Vehicle Speed Estimation Using a Single Camera and Computer Vision Techniques |
| 3 | Development tool for image enhancement and image filter in spatial domain |
| 4 | Real-time license plate recognition using yolo26n and DIPR techniques |
| 5 | AI Smart Home Security and Automation System Using Computer Vision |
| 6 | Traffic Sign Detection and Classification Using Image Processing and YOLO |
| 7 | Vehicle License Plate Recognition Using Image Processing Techniques |

Khi chọn đề mới, **tránh trùng hoặc quá gần** các đề trong bảng trên (theo hướng dẫn của GV).

---

## So khớp với pool 24 đề (danh sách bạn gửi)

**Loại vì trùng / quá gần đề đã đăng ký (7 nhóm):**

| Đề trong pool 24 | Lý do loại |
|------------------|------------|
| Vehicle Speed Estimation Using a Single Camera and Computer Vision Techniques | Trùng nhóm 2 |
| Traffic Sign Detection and Classification Using Image Processing and YOLO | Trùng nhóm 6 |
| Automatic Number Plate Recognition for an Automated Parking Fee System | Trùng hướng biển số (nhóm 4, 7) |
| License Plate Detection Using Image Processing Techniques | Trùng hướng biển số (nhóm 4, 7) |
| Real-Time Hand Gesture Recognition for Game Control | Quá gần nhóm 1 (gesture / điều khiển) |

**Còn có thể chọn (19 đề):**

1. Attendance System Using Face Recognition and Embedded Camera  
2. Automatic Red-Light Violation Detection System Using Computer Vision  
3. Autonomous Delivery Robot Using Vision-Based Navigation  
4. Car Detection Using Image Processing and Pattern Recognition  
5. Classification of Student Learning Engagement in Class Using Facial Analysis  
6. Design of an Auto-Tracking Tripod Using Face Recognition  
7. Driver Drowsiness Detection System Using Computer Vision  
8. Face Recognition-Based Smart Door Lock System Using Raspberry Pi  
9. Fruit Variety Classification Using Image Processing Techniques  
10. Human Action Recognition in Video Using YOLO-Based Skeleton Extraction  
11. Line-Following Robot with Vision-Based Path Detection  
12. Object Following Robot Using Camera-Based Tracking  
13. Object Recognition Using Image Processing Techniques  
14. Object Tracking Robot Using Computer Vision and Embedded Camera  
15. People Counting System Using a Fixed Surveillance Camera  
16. Real-Time Helmet Detection System for Motorcyclists Using a Camera  
17. Real-Time Mask Detection System with Embedded Camera  
18. Smart Parking Space Detection Using Overhead Camera  
19. Smart Traffic Light Control System Using Vehicle Detection  

*(Cuối cùng vẫn xin xác nhận với GV.)*

---

## Đề xuất tối ưu (ưu tiên môn DIP + khả thi)

**Đề chính:** *Fruit Variety Classification Using Image Processing Techniques*

- Bám **DIP rõ**: phân đoạn / ROI, đặc trưng màu-kết cấu (histogram, LBP, HOG…), phân loại (k-NN, SVM), có thể so sánh với CNN như phần mở rộng.  
- **Không đụng** các nhóm biển số / biển báo / tốc độ / gesture.  
- Dữ liệu: tự chụp hoặc tập mở (ít phụ thuộc robot / nhiều thiết bị đắt).

**Dự phòng:** *People Counting System Using a Fixed Surveillance Camera* (nền tĩnh, dễ demo) hoặc *Driver Drowsiness Detection System Using Computer Vision* (ứng dụng rõ, cần webcam tốt).

---

## Chuẩn bị thiết bị (gợi ý)

### Nếu chọn Fruit Variety Classification
- **Máy tính:** Windows/Linux, Python + OpenCV; GPU tùy chọn (nếu sau này thử deep learning).  
- **Camera:** webcam hoặc điện thoại chụp cố định (đèn đều, nền đơn giản).  
- **Phụ kiện:** giấy trắng / nền một màu làm nền chụp; tùy chọn thẻ màu để hiệu chỉnh.  
- **Không bắt buộc:** Raspberry Pi (chỉ khi muốn demo nhúng).

### Nếu chọn People Counting
- **Camera cố định** nhìn xuống / chéo cửa (USB webcam hoặc IP camera nếu có).  
- PC chạy real-time nhẹ (OpenCV + tracker / background subtraction).

### Nếu chọn Driver Drowsiness
- **Webcam 720p+**, ưu tiên có tự động lấy nét; tùy chọn webcam hồng ngoại nếu demo thiếu sáng.  
- Màn hình đặt ngang tầm mắt khi quay demo.

### Nếu chọn đề robot (line-following / object tracking / delivery…)
- **Xe robot** + **camera module** (Pi Camera / USB cam) + **vi điều khiển** (Arduino / Raspberry Pi / Jetson tùy ngân sách).  
- Pin, khung, driver motor; dự phòng thời gian hiệu chỉnh cơ khí + ánh sáng.

```powershell
pip install -r requirements.txt
```

---

## Đồ án đang làm: Fruit classification

Code trong thư mục `fruit_project/`. Hướng dẫn điện thoại làm webcam: **`README_FRUIT.md`**.
