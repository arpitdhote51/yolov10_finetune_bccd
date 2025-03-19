import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained model
model = YOLO("yolov10_bccd.pt")  # Ensure this file exists

# Function to perform object detection
import cv2

def detect_cells(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image, conf=0.1)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    return output_path


# Create Gradio Interface
app = gr.Interface(
    fn=detect_cells, 
    inputs=gr.Image(type="pil"),  # Upload image
    outputs=gr.Image(type="numpy"),  # Show processed image
    title="Blood Cell Detection with YOLOv10",
    description="Upload an image to detect RBCs, WBCs, and Platelets."
)

# Launch the app
if __name__ == "__main__":
    app.launch()
