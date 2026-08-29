from ultralytics import YOLO

model = YOLO("yolo26x.pt")

results = model.predict(source="test_img/1.png", save=True, show=True)

print("done!")