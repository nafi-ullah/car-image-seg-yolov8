from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLOv8 segmentation model
model_path = 'best.pt'
image_path = 'carimage.jpg'
output_path = 'carimage_masked.png'

# Define colors for each part (BGR format)
color_mapping = {
    0: (255, 255, 255),  # background: white
    1: (0, 0, 0),        # car: black
    2: (0, 255, 0),      # wheel: green
    3: (0, 0, 255),      # headlight: red
    4: (0, 255, 255)     # window: yellow
}

# Read the input image
img = cv2.imread(image_path)
H, W, _ = img.shape

# Load the model
model = YOLO(model_path)

# Perform inference on the image
results = model(img)

# Initialize a blank mask image with a white background
output_mask = np.full((H, W, 3), color_mapping[0], dtype=np.uint8)  # Start with white background

# Process each detected object and apply corresponding color masks
for result in results:
    for class_id, mask in enumerate(result.masks.data):
        mask = mask.numpy()  # Convert mask to numpy array
        mask_resized = cv2.resize(mask, (W, H))  # Resize mask to match original image size
        
        # Apply the mask color based on the class ID
        color = color_mapping.get(class_id + 1, (0, 0, 0))  # Default to black if class_id is out of range
        output_mask[mask_resized > 0.5] = color  # Use 0.5 as threshold for mask

# Save the output mask image
cv2.imwrite(output_path, output_mask)

print(f"Masked image saved as {output_path}")
