# 代码示例：JSON转YOLO格式（目标检测）
import json
import os

def convert_labelme_to_yolo(json_path, output_dir, label_map):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    txt_lines = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        x_min, y_min = min(p[0] for p in points), min(p[1] for p in points)
        x_max, y_max = max(p[0] for p in points), max(p[1] for p in points)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        txt_lines.append(f"{label_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    txt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
