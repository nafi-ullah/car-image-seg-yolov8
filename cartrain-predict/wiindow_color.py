from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLOv8 segmentation model
model_path = 'best.pt'
image_path = 'carimage.jpg'
output_path = 'carimage_window_masked.png'

# Define colors for the mask (BGR format)
color_mapping = {
    'default': (255, 255, 255),  # default: white
    'window': (0, 0, 0),         # window: black
}

# Read the input image
img = cv2.imread(image_path)
H, W, _ = img.shape

# Load the model
model = YOLO(model_path)

# Perform inference on the image
results = model(img)

# Initialize a blank mask image with a white background
output_mask = np.full((H, W, 3), color_mapping['default'], dtype=np.uint8)  # Start with white background

# Process each detected object and apply mask only for windows
for result in results:
    for class_id, mask in enumerate(result.masks.data):
        mask = mask.numpy()  # Convert mask to numpy array
        mask_resized = cv2.resize(mask, (W, H))  # Resize mask to match original image size
        
        if class_id == 3:  # Assuming class_id 3 corresponds to 'window'
            output_mask[mask_resized > 0.5] = color_mapping['window']  # Apply black color to window areas

# Save the output mask image
cv2.imwrite(output_path, output_mask)

print(f"Masked image saved as {output_path}")
