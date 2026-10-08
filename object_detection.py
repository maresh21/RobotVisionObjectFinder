import cv2
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------------
# 1. TRAIN KNN MODEL
# -----------------------------------

X = []
y = []

with open("dataset/objects.csv", "r") as file:
    reader = csv.reader(file)

    for row in reader:
        label = row[0]

        features = [
            int(row[1]),
            int(row[2]),
            int(row[3])
        ]

        X.append(features)
        y.append(label)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print("KNN model trained successfully!")


# -----------------------------------
# 2. LOAD IMAGE
# -----------------------------------

image = cv2.imread("test.jpg")

if image is None:
    print("ERROR: test.jpg not found!")
    exit()


# -----------------------------------
# 3. CONVERT IMAGE TO HSV
# -----------------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# -----------------------------------
# 4. DEFINE COLOR RANGES
# -----------------------------------

colors = {
    "red": (
        np.array([0, 100, 100]),
        np.array([10, 255, 255])
    ),

    "blue": (
        np.array([100, 100, 100]),
        np.array([140, 255, 255])
    ),

    "green": (
        np.array([40, 50, 50]),
        np.array([80, 255, 255])
    )
}


# -----------------------------------
# 5. DETECT OBJECTS
# -----------------------------------

detected_objects = []


for color_name, (lower, upper) in colors.items():

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 500:

            x, y_pos, w, h = cv2.boundingRect(contour)

            # -----------------------------------
            # 6. EXTRACT FEATURES
            # -----------------------------------

            feature_1 = 1 if w > h else 0
            feature_2 = 1 if h > w else 0
            feature_3 = 1 if area > 2000 else 0

            features = [[
                feature_1,
                feature_2,
                feature_3
            ]]


            # -----------------------------------
            # 7. KNN CLASSIFICATION
            # -----------------------------------

            prediction = model.predict(features)

            predicted_label = prediction[0]


            # -----------------------------------
            # 8. SAVE DETECTED OBJECT
            # -----------------------------------

            detected_objects.append(predicted_label)

            # Draw rectangle
            cv2.rectangle(
                image,
                (x, y_pos),
                (x + w, y_pos + h),
                (0, 255, 0),
                2
            )

            # Display classification
            cv2.putText(
                image,
                predicted_label,
                (x, y_pos - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


# -----------------------------------
# 9. SAVE RESULT
# -----------------------------------

cv2.imwrite("classified_result.jpg", image)


# -----------------------------------
# 10. DISPLAY RESULT
# -----------------------------------

print()
print("Object detection completed!")

if detected_objects:
    print("Detected objects:")

    for obj in detected_objects:
        print("-", obj)

else:
    print("No objects detected.")

print()
print("Result saved as: classified_result.jpg")