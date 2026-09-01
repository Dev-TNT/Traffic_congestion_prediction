import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import core.Get_frame as Get_frame

model = YOLO("yolo26x.pt")
# image = cv2.imread("test_img/crowded_traffic.jpeg")
image = Get_frame.get_traffic_image("Camera 1")


def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(callback=callback, slice_wh=(512,512))
detections = slicer(image)

# Đếm theo class ID của YOLO
class_ids = detections.class_id

person_count = np.sum(class_ids == 0)
car_count = np.sum(class_ids == 2)
motorbike_count = np.sum(class_ids == 3)
bus_count = np.sum(class_ids == 5)
truck_count = np.sum(class_ids == 7)
total_count = len(detections)

print(f"Person: {person_count}")
print(f"Car: {car_count}")
print(f"Motorbike: {motorbike_count}")
print(f"Bus: {bus_count}")
print(f"Truck: {truck_count}")
print(f"Total objects: {total_count}")


box_annotator = sv.BoxAnnotator()

annotator_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

cv2.imwrite("runs/supervision_test9.png", annotator_image)

