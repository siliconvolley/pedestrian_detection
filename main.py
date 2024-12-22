from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Load YOLO model using Ultralytics
model = YOLO('yolov8n.pt')

# Class names for the YOLO model (ensure these match your model's training classes)
class_names = model.names

# Main pedestrian detection function
def detect_pedestrians(image):
    # Run YOLO detection
    results = model(image)

    # Process detections
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

    return detections, confidences, class_ids

# Draw bounding boxes on image
def draw_detections(image, detections, confidences, class_ids):
    for box, confidence, class_id in zip(detections, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(class_id)]}: {confidence:.2f}"  # Use the detected class name

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Process a set of images and plot the results
def process_images(image_dir, output_dir):
    all_detections = []
    os.makedirs(output_dir, exist_ok=True)

    image_files = os.listdir(image_dir)
    processed_images = []

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        detections, confidences, class_ids = detect_pedestrians(image)
        all_detections.append(detections)

        # Draw detections and save the image
        image_with_detections = draw_detections(image.copy(), detections, confidences, class_ids)
        processed_images.append(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(output_dir, f"detected_{image_name}"), image_with_detections)

    plot_images(processed_images)

    cv2.destroyAllWindows()

def plot_images(processed_images):
    # Plot all processed images in a single Matplotlib figure
    num_images = len(processed_images)
    cols = 3  # Number of columns in the plot grid
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * rows))
    for i, img in enumerate(processed_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    image_directory = "./images"  # Directory with input images
    output_directory = "./output"  # Directory for output images

    process_images(image_directory, output_directory)
