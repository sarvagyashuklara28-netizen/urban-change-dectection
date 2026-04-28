import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('urban_image.jpg')

# Check image
if image is None:
    print("Error: Image not found.")
    exit()

# Resize image
image = cv2.resize(image, (800, 600))

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# --------------------------
# Vegetation Detection
# --------------------------
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)

# --------------------------
# Water Detection
# --------------------------
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

water_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# --------------------------
# Road / Gray Area Detection
# --------------------------
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 50, 200])

road_mask = cv2.inRange(hsv, lower_gray, upper_gray)

# --------------------------
# Building Detection
# --------------------------
lower_building = np.array([0, 0, 0])
upper_building = np.array([180, 255, 50])

building_mask = cv2.inRange(hsv, lower_building, upper_building)

# --------------------------
# Pixel Calculations
# --------------------------
total_pixels = image.shape[0] * image.shape[1]

vegetation_pixels = cv2.countNonZero(vegetation_mask)
water_pixels = cv2.countNonZero(water_mask)
road_pixels = cv2.countNonZero(road_mask)
building_pixels = cv2.countNonZero(building_mask)

vegetation_percent = (vegetation_pixels / total_pixels) * 100
water_percent = (water_pixels / total_pixels) * 100
road_percent = (road_pixels / total_pixels) * 100
building_percent = (building_pixels / total_pixels) * 100

# --------------------------
# Display Results
# --------------------------
print("Urban Change Detection Results")
print("--------------------------------")
print(f"Vegetation Area : {vegetation_percent:.2f}%")
print(f"Water Area      : {water_percent:.2f}%")
print(f"Road Area       : {road_percent:.2f}%")
print(f"Building Area   : {building_percent:.2f}%")

# --------------------------
# Show Images
# --------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(vegetation_mask, cmap='Greens')
plt.title('Vegetation Detection')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(water_mask, cmap='Blues')
plt.title('Water Detection')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(road_mask, cmap='gray')
plt.title('Road Detection')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(building_mask, cmap='inferno')
plt.title('Building Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

# Required Libraries

Open terminal in VS Code and run:

```bash
pip install opencv-python numpy matplotlib
```

---

# How to Run

1. Save aerial image as:

```text
urban_image.jpg
```

2. Put the image in the same folder as `main.py`

3. Open terminal in VS Code.

4. Run:

```bash
python main.py
```

---

# Suggested Folder Structure

```text
Urban-Change-Detection/
│
├── main.py
├── urban_image.jpg
├── README.md
├── results/
└── screenshots/
```

---

# What This Program Does

* Detects vegetation
* Detects water bodies
* Detects roads
* Detects buildings
* Calculates percentage area occupied
* Displays segmented output images
* Supports urban change detection and encroachment monitoring
