import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import pandas as pd

data = pd.read_csv("BCCD_Dataset-master/test.csv")

def get_true_labels(image_name):
    """Get true labels for the given image as a list of class IDs."""
    true_boxes = data[data['filename'] == image_name]
    
    true_labels = []
    for index, row in true_boxes.iterrows():
        # Mapping cell types to class IDs (WBC: 0, RBC: 1, Platelets: 2)
        if row['cell_type'] == 'WBC':
            true_labels.append(0)
        elif row['cell_type'] == 'RBC':
            true_labels.append(1)
        elif row['cell_type'] == 'Platelets':
            true_labels.append(2)
    
    return true_labels


model = YOLO('best (1).pt')

class_names = {0: "WBC", 1: "RBC", 2: "Platelets"}
class_colors = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0)
}

def inference(image, image_name):
    """Perform inference and return detections, true labels, and predicted labels."""
    results = model(image)[0]
    detections = []
    true_labels = get_true_labels(image_name)
    predicted_labels = []
    
    for box in results.boxes:
        box_xyxy = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls.cpu().numpy())
        confidence = float(box.conf.cpu().numpy())
        
        if confidence > 0.5:  # Confidence threshold
            detections.append({
                "bbox": box_xyxy,
                "class_id": class_id,
                "confidence": confidence
            })
            predicted_labels.append(class_id)

    return detections, true_labels, predicted_labels

def display(image, image_name):
    """Display the image with detections and return processed image with metrics."""
    detections, true_labels, predicted_labels = inference(image, image_name)
    image_display = image.copy()

    # Drawing detections
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        class_id = detection["class_id"]
        class_name = class_names.get(class_id, "Unknown")
        color = class_colors.get(class_id, (255, 255, 255))

        cv2.rectangle(image_display, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {detection['confidence']:.2f}"
        cv2.putText(image_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    # Calculating Metrics
    precision = recall = 0.0
    if len(predicted_labels) > 0 and len(true_labels) > 0:
        max_length = max(len(predicted_labels), len(true_labels))
        predicted_labels_padded = predicted_labels + [-1] * (max_length - len(predicted_labels))
        true_labels_padded = true_labels + [-1] * (max_length - len(true_labels))
        
        try:
            precision = precision_score(true_labels_padded, predicted_labels_padded, average='weighted', zero_division=0)
            recall = recall_score(true_labels_padded, predicted_labels_padded, average='weighted', zero_division=0)
            print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            precision = recall = 0.0

    return image_rgb, precision, recall

