from ultralytics import YOLO
from PIL import Image
# from IPython.display import Image
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import locale
import os

locale.getpreferredencoding = lambda: "UTF-8"

# Load a sample image from the dataset to see if it works on the untrained model
validation_img_path = 'E:/work/github/bovinae/yolo11_test/content/datasets/brain-tumor/valid/images/val_1 (1).jpg'
brain_img = Image.open(validation_img_path)

os.chdir(r"E:\work\github\bovinae\yolo11_test")
model_n = YOLO('yolo11n.pt')
model_s = YOLO('yolo11s.pt')
model_m = YOLO('yolo11m.pt')
model_l = YOLO('yolo11l.pt')
model_x = YOLO('yolo11x.pt')

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(brain_img)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Prepare model names and grid positions (excluding (0, 0))
models = [model_n, model_s, model_m, model_l, model_x]
model_names = ['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x']
positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # Skip (0, 0)

# Loop through models and plot their results
for model, name, (row, col) in zip(models, model_names, positions):    
    results = model(validation_img_path)    
    result_img = results[0].plot()
    
    axes[row, col].imshow(result_img)    
    axes[row, col].set_title(f'{name} Detection Results')    
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()