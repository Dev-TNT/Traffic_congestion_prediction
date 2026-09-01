import cv2
import time
import core.Get_frame as Get_frame
import core.Yolo_detect as Yolo_detect

def main():
    window_name = "Traffic Camera AI Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 450)

    while True:
        start_t = time.time()
        frame = Get_frame.get_traffic_image("Camera 1")
        if frame is not None:
            (annotated_frame,
             person_count,
             car_count,
             motorbike_count,
             bus_count,
             truck_count,
             total_vehicles) \
                = Yolo_detect.image_processing(frame)

            print(f"Detected - "
                  f"Persons: {person_count}, "
                  f"Cars: {car_count}, "
                  f"Motorbikes: {motorbike_count}, "
                  f"Buses: {bus_count}, "
                  f"Trucks: {truck_count}, "
                  f"Total Vehicles: {total_vehicles}")

            cv2.imshow(window_name, annotated_frame)

        elapsed_time = time.time() - start_t
        sleep_time = max(1.0, 10.0 - elapsed_time)

        if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()