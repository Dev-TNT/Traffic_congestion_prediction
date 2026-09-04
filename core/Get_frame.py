import requests
import cv2
import numpy as np
import time

def get_traffic_image(name="Camera 1"):
    global cam_id
    timestamp = int(time.time() * 1000)
    url = f"https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id={cam_id[name]}&t={timestamp}"

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
                print(f"Server response error: {response.status_code}")
    except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")

    return None




cam_id = {
    "Camera 1": "5d9ddd49766c880017188c94",
    "Camera 2": "5d9ddec9766c880017188c9c",
    "Camera 3": "5b0e1faacddcc80011ceb449",
    "Camera 4": "5d9ddf49766c880017188ca0",
    "Camera 5": "5d9dde1f766c880017188c98"
}
print(f"Starting data collection loop for camera: {list(cam_id.keys())}...")

if __name__ == "__main__":
    while True:
        frame = get_traffic_image()
        if frame is not None:
            cv2.imshow("Traffic Camera Feed", frame)
        else:
                print("No image received from the camera.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
