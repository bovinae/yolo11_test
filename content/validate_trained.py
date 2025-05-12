import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === Load trained YOLO model ==
model_path=r'E:\work\github\bovinae\yolo11_test\runs\detect\train\weights\best.pt'
model =YOLO(model_path)

# === List of image-label pairs ===
pairs = [
    (r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\images\val_1 (1).jpg",
    r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\labels\val_1 (1).txt"),

    (r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\images\val_1 (2).jpg",
    r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\labels\val_1 (2).txt"),

    (r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\images\val_1 (3).jpg",
    r"E:\work\github\bovinae\yolo11_test\content\datasets\brain-tumor\valid\labels\val_1 (3).txt")
]

# === Loop through each pair ===
for img_path, label_path in pairs:
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_h, img_w, _= image.shape

    # Copy for ground truth box drawing
    ground_truth_img = image.copy()

    # Load label(s) and draw
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center-width / 2) * img_w)
            y1 = int((y_center-height / 2) * img_h)
            x2 = int((x_center + width/2) * img_w)
            y2 = int((y_center + height /2) * img_h)

            label_name = "positive" if int(class_id) == 1 else "negative"
            cv2.rectangle(ground_truth_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(ground_truth_img, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Model prediction
    results = model(img_path)
    prediction_img =results[0].plot()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(ground_truth_img)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    ax2.imshow(prediction_img)
    ax2.set_title('Model Prediction')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
