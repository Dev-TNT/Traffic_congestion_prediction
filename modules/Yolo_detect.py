from ultralytics import YOLO
from datetime import datetime
import cv2
import supervision as sv
import numpy as np

model = YOLO("yolo26x.pt")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

def image_processing(frame):
    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(300, 300))
    detections = slicer(frame)

    # Đếm theo class ID của YOLO
    class_ids = detections.class_id

    person_count = np.sum(class_ids == 0)
    car_count = np.sum(class_ids == 2)
    motorbike_count = np.sum(class_ids == 3)
    bus_count = np.sum(class_ids == 5)
    truck_count = np.sum(class_ids == 7)
    total_vehicles = len(detections)

    print(f"Person: {person_count}")
    print(f"Car: {car_count}")
    print(f"Motorbike: {motorbike_count}")
    print(f"Bus: {bus_count}")
    print(f"Truck: {truck_count}")
    print(f"Total vehicles: {total_vehicles}")

    box_annotator = sv.BoxAnnotator()

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, f"{current_time}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return annotated_frame, person_count, car_count, motorbike_count, bus_count, truck_count, total_vehicles


