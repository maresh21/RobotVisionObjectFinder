import cv2
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier

from navigation import decide_movement


# ==========================================
# 1. TRAIN KNN MODEL
# ==========================================

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


# ==========================================
# 2. LOAD IMAGE
# ==========================================

image = cv2.imread("test.jpg")

if image is None:
    print("ERROR: test.jpg not found!")
    exit()


image_height, image_width = image.shape[:2]

print("Image size:", image_width, "x", image_height)


# ==========================================
# 3. CONVERT IMAGE TO HSV
# ==========================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ==========================================
# 4. COLOR RANGES
# ==========================================

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


# ==========================================
# 5. FIND OBJECTS
# ==========================================

objects = []


for color_name, (lower, upper) in colors.items():

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # ==========================================
        # 6. EXTRACT FEATURES
        # ==========================================

        feature_1 = 1 if w > h else 0
        feature_2 = 1 if h > w else 0
        feature_3 = 1 if area > 2000 else 0

        features = [[
            feature_1,
            feature_2,
            feature_3
        ]]


        # ==========================================
        # 7. KNN CLASSIFICATION
        # ==========================================

        prediction = model.predict(features)

        label = prediction[0]


        # ==========================================
        # 8. FIND OBJECT CENTER
        # ==========================================

        object_center_x = x + w // 2
        object_center_y = y + h // 2


        # ==========================================
        # 9. NAVIGATION DECISION
        # ==========================================

        action = decide_movement(
            object_center_x,
            image_width,
            area
        )


        # ==========================================
        # 10. STORE OBJECT
        # ==========================================

        objects.append({
            "label": label,
            "x": object_center_x,
            "y": object_center_y,
            "area": area,
            "action": action
        })


        # ==========================================
        # 11. DRAW OBJECT
        # ==========================================

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            image,
            (object_center_x, object_center_y),
            5,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            image,
            label,
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            action,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print()
print("========== ROBOT VISION RESULTS ==========")

if objects:

    for i, obj in enumerate(objects):

        print()
        print("Object", i + 1)
        print("Class:", obj["label"])
        print("Position:", obj["x"], ",", obj["y"])
        print("Area:", obj["area"])
        print("Robot action:", obj["action"])

else:

    print("No objects detected.")


print()
print("===========================================")


# ==========================================
# 13. SAVE IMAGE
# ==========================================

cv2.imwrite(
    "robot_navigation_result.jpg",
    image
)

print()
print("Result saved as: robot_navigation_result.jpg")