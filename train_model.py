import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Path to your gesture folders
DATA_PATH = "Data"

X = []
y = []

# Loop over each label folder
for label in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, label)
    
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img_resized = cv2.resize(img, (100, 100))  # Resize all images
            X.append(img_resized.flatten())            # Flatten the image
            y.append(label)

# Convert to NumPy array
X = np.array(X)
y = np.array(y)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
joblib.dump(model, "asl_knn_model.pkl")
print("Model trained and saved as 'asl_knn_model.pkl'")
