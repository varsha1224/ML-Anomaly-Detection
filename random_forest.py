import os
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


json_file = "input.json"
with open(json_file) as f:
    data = json.load(f)
    dieWidth = data['die']['width']
    dieHeight = data['die']['height']
    streetWidth = data['street_width']
    careAreas = data['care_areas']
    exclusionZones = data.get('exclusion_zones')

def label_regions(image, care_areas, exclusion_zones):
    labels = np.zeros_like(image, dtype=np.uint8)  
    
    for area in care_areas:
        x1, y1 = area['top_left']['x'], area['top_left']['y']
        x2, y2 = area['bottom_right']['x'], area['bottom_right']['y']
        labels[y1:y2, x1:x2] = 1  
    
    if exclusion_zones:
        for zone in exclusion_zones:
            x1, y1 = zone['top_left']['x'], zone['top_left']['y']
            x2, y2 = zone['bottom_right']['x'], zone['bottom_right']['y']
            labels[y1:y2, x1:x2] = 0  
    return labels


def extract_features(image):
    # Extract mean and standard deviation of pixel values as features
    mean = np.mean(image)
    std_dev = np.std(image)
    return [mean, std_dev]


image_directory = "wafer_image/"

features = []
labels = []

def is_inside_care_area(image, care_areas):
    care_area = care_areas[0]
    x1, y1 = care_area['top_left']['x'], care_area['top_left']['y']
    x2, y2 = care_area['bottom_right']['x'], care_area['bottom_right']['y']
    
    # Check if the center of the image (assuming square image) is inside the care area
    image_center_x = image.shape[1] // 2
    image_center_y = image.shape[0] // 2
    
    return x1 <= image_center_x <= x2 and y1 <= image_center_y <= y2

for filename in os.listdir(image_directory):
    if filename.endswith('.png') and filename.startswith('wafer_image_'):
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)        
        feature_vector = extract_features(image)
        
        # Label the entire image based on whether it's inside the care area or not
        label = 1 if is_inside_care_area(image, careAreas) else 0   
        labels.append(label)
        features.append(feature_vector)


print("Features:\n", features)
print(len(labels))


X = np.array(features)
y = np.array(labels)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
print("predicted value:",y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)


plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Anomalies', color='red')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Normal', color='blue')

plt.xlabel('Mean')
plt.ylabel('Standard Deviation')
plt.legend()
plt.title('Feature Visualization')
plt.show()

