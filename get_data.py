import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from core.Get_frame import get_traffic_image, cam_id
from core.Yolo_detect import image_processing

DEFAULT_OUTPUT = "data.csv"
DEFAULT_CAMERA = "Camera 1"
DEFAULT_MAX_SECONDS = 24 * 60 * 60


def save_rows_to_csv(rows, output_path):
    fieldnames = ["timestamp", "person", "car", "motorbike", "bus", "truck", "total_vehicles"]
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã lưu dữ liệu vào: {output_path}")


def collect_data(camera_name=DEFAULT_CAMERA, output_path=DEFAULT_OUTPUT, max_seconds=DEFAULT_MAX_SECONDS):
    rows = []
    last_signature = None
    last_annotated_frame = None
    start_time = time.time()
    window_name = "Traffic YOLO Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Bắt đầu thu thập dữ liệu từ {camera_name}")
    print(f"Tự động dừng sau {max_seconds / 3600:.2f} giờ hoặc khi nhấn 'q'.")
    print(f"File lưu: {output_path}")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                print("Đã chạy đủ 24 tiếng. Kết thúc thu thập dữ liệu.")
                break

            frame = get_traffic_image(camera_name)
            if frame is not None:
                signature = frame.tobytes()

                if signature == last_signature:
                    if last_annotated_frame is not None:
                        cv2.imshow(window_name, last_annotated_frame)
                    else:
                        cv2.imshow(window_name, frame)
                else:
                    last_signature = signature
                    annotated_frame, person, car, motorbike, bus, truck, total = image_processing(frame)
                    last_annotated_frame = annotated_frame
                    cv2.imshow(window_name, annotated_frame)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    rows.append({
                        "timestamp": timestamp,
                        "person": int(person),
                        "car": int(car),
                        "motorbike": int(motorbike),
                        "bus": int(bus),
                        "truck": int(truck),
                        "total_vehicles": int(total),
                    })
            else:
                print("Không nhận được hình ảnh từ camera. Đang thử lại...")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Người dùng nhấn 'q'. Kết thúc thu thập dữ liệu.")
                break

    finally:
        cv2.destroyAllWindows()

    save_rows_to_csv(rows, output_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu từ camera YOLO trong 24 tiếng hoặc cho đến khi nhấn q.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Tên camera cần detect. Mặc định: Camera 1")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Đường dẫn file CSV để lưu dữ liệu. Mặc định: data.csv")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS, help="Thời gian tối đa chạy (giây). Mặc định: 86400")
    args = parser.parse_args()

    if args.camera not in cam_id:
        raise ValueError(f"Camera '{args.camera}' không tồn tại. Các camera có sẵn: {list(cam_id.keys())}")

    collect_data(camera_name=args.camera, output_path=args.output, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
