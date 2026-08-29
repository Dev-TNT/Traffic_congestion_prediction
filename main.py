import cv2
import numpy as np
import requests
import time
from datetime import datetime
from ultralytics import YOLO

# Khởi tạo mô hình bản X (Độ chính xác cao nhất, xử lý chậm nhất)
model = YOLO("yolo26x.pt")


def get_traffic_image(cam_id):
    timestamp = int(time.time() * 1000)
    url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id}&t={timestamp}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://giaothong.hochiminhcity.gov.vn/Map.aspx"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        else:
            print(f"Lỗi phản hồi từ server: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi đường truyền: {e}")

    return None


def main():
    CAMERA_ID = "56df81d8c062921100c143de"
    print(f"Khởi động luồng thu thập dữ liệu cho Camera: {CAMERA_ID}...")

    window_name = "Traffic Camera AI Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 450)

    while True:
        start_t = time.time()
        frame = get_traffic_image(CAMERA_ID)

        if frame is not None:
            # 1. Chạy dự đoán trên frame lấy về.
            results = model.predict(source=frame, conf=0.4, classes=[2, 3, 5, 7], verbose=False)

            # 2. Lấy frame đã được AI vẽ sẵn các ô vuông (bounding boxes)
            annotated_frame = results[0].plot()

            # --- PHẦN TÍCH HỢP ĐẾM VÀ HIỂN THỊ THÔNG SỐ ---

            # Trích xuất danh sách ID nhãn (class) và đếm tổng
            detected_classes = results[0].boxes.cls.cpu().tolist()
            total_vehicles = len(detected_classes)

            # Đếm chi tiết từng loại xe dựa trên ID của COCO dataset
            car_count = detected_classes.count(2)
            motorbike_count = detected_classes.count(3)
            bus_count = detected_classes.count(5)
            truck_count = detected_classes.count(7)

            # --- VẼ CHỮ LÊN MÀN HÌNH ---
            # a. Gắn timestamp (Góc trên trái)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, f"{current_time}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # b. Gắn chi tiết từng loại xe (Nằm dưới timestamp)
            details_text = f"Car: {car_count} | Bike: {motorbike_count} | Bus: {bus_count} | Truck: {truck_count}"
            cv2.putText(annotated_frame, details_text, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # c. Gắn tổng số phương tiện (Góc trên bên phải)
            cv2.putText(annotated_frame, f"TOTAL: {total_vehicles}", (600, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # =========================================================

            # 4. Hiển thị frame cuối cùng lên màn hình
            cv2.imshow(window_name, annotated_frame)

        elapsed_time = time.time() - start_t
        sleep_time = max(1.0, 10.0 - elapsed_time)

        if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()