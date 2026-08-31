from ultralytics import YOLO
from datetime import datetime
import cv2

model = YOLO("yolo26x.pt")

def image_processing(frame):
    # 1. Chạy dự đoán trên frame lấy về.
    results = model.predict(source=frame, conf=0.4, classes=[2, 3, 5, 7], verbose=False)

    # 2. Lấy frame đã được AI vẽ sẵn các ô vuông (bounding boxes)
    annotated_frame = results[0].plot()

    # --- PHẦN TÍCH HỢP ĐẾM VÀ HIỂN THỊ THÔNG SỐ ---

    # Trích xuất danh sách ID nhãn (class) và đếm tổng
    detected_classes = results[0].boxes.cls.cpu().tolist()
    total_vehicles = len(detected_classes)

    # Đếm chi tiết từng loại xe dựa trên ID của COCO dataset
    person_count = detected_classes.count(0)
    car_count = detected_classes.count(2)
    motorbike_count = detected_classes.count(3)
    bus_count = detected_classes.count(5)
    truck_count = detected_classes.count(7)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, f"{current_time}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return annotated_frame, person_count, car_count, motorbike_count, bus_count, truck_count, total_vehicles


