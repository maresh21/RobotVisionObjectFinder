import cv2
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# 1. TRAIN KNN MODEL
# ==========================================

X = []
y = []

with open("dataset/objects.csv", "r") as file:

    reader = csv.reader(file)

    for row in reader:

        label = row[0]

        hue = float(row[1])
        corners = int(row[2])

        X.append([
            hue,
            corners
        ])

        y.append(label)


model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

print()
print("KNN model trained successfully!")


# ==========================================
# 2. LOAD IMAGE
# ==========================================

image = cv2.imread("test.jpg")

if image is None:

    print("ERROR: test.jpg not found!")

    exit()


height, width = image.shape[:2]


# ==========================================
# 3. CONVERT IMAGE TO HSV
# ==========================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ==========================================
# 4. COLOR MASKS
# ==========================================

color_ranges = {

    "red": [
        np.array([0, 100, 100]),
        np.array([10, 255, 255])
    ],

    "green": [
        np.array([35, 50, 50]),
        np.array([85, 255, 255])
    ],

    "blue": [
        np.array([90, 50, 50]),
        np.array([140, 255, 255])
    ]
}


# ==========================================
# 5. DETECT ALL OBJECTS
# ==========================================

detected_objects = []


for color_name, ranges in color_ranges.items():

    lower = ranges[0]
    upper = ranges[1]

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        area = cv2.contourArea(contour)


        # Ignore tiny objects/noise

        if area < 500:

            continue


        # ==========================================
        # FIND BOUNDING BOX
        # ==========================================

        x, y, w, h = cv2.boundingRect(contour)


        # ==========================================
        # FIND CENTER
        # ==========================================

        center_x = x + w // 2
        center_y = y + h // 2


        # ==========================================
        # FIND SHAPE
        # ==========================================

        perimeter = cv2.arcLength(
            contour,
            True
        )


        if perimeter == 0:

            continue


        approximation = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )


        corners = len(approximation)


        # ==========================================
        # LIMIT CORNERS FOR CIRCLES
        # ==========================================

        if corners > 7:

            shape_corners = 8

        else:

            shape_corners = corners


        # ==========================================
        # FIND HUE
        # ==========================================

        object_mask = mask[y:y+h, x:x+w]

        object_hsv = hsv[y:y+h, x:x+w]

        hue_values = object_hsv[:, :, 0][
            object_mask > 0
        ]


        if len(hue_values) == 0:

            continue


        hue = float(
            np.mean(hue_values)
        )


        # ==========================================
        # KNN CLASSIFICATION
        # ==========================================

        features = [[
            hue,
            shape_corners
        ]]


        prediction = model.predict(
            features
        )


        object_name = prediction[0]


        # ==========================================
        # NAVIGATION
        # ==========================================

        image_center = width // 2


        if area > 15000:

            action = "STOP"

        elif center_x < image_center - 100:

            action = "TURN LEFT"

        elif center_x > image_center + 100:

            action = "TURN RIGHT"

        else:

            action = "MOVE FORWARD"


        # ==========================================
        # SAVE OBJECT
        # ==========================================

        detected_objects.append({

            "name": object_name,

            "color": color_name,

            "x": center_x,

            "y": center_y,

            "area": area,

            "action": action

        })


        # ==========================================
        # DRAW RESULT
        # ==========================================

        cv2.rectangle(
            image,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            3
        )


        cv2.circle(
            image,

            (center_x, center_y),

            6,

            (255, 0, 0),

            -1
        )


        cv2.putText(

            image,

            object_name,

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
# 6. PRINT RESULTS
# ==========================================

print()
print("========================================")
print("       ROBOT VISION OBJECT FINDER")
print("========================================")


if len(detected_objects) == 0:

    print("No objects detected.")


else:

    for i, obj in enumerate(
        detected_objects
    ):

        print()

        print(
            "Object",
            i + 1
        )

        print(
            "Class       :",
            obj["name"]
        )

        print(
            "Color       :",
            obj["color"]
        )

        print(
            "Position    :",
            obj["x"],
            ",",
            obj["y"]
        )

        print(
            "Area        :",
            round(
                obj["area"],
                2
            )
        )

        print(
            "Robot Action:",
            obj["action"]
        )


print()
print("========================================")


# ==========================================
# 7. SAVE OUTPUT IMAGE
# ==========================================

cv2.imwrite(
    "final_result.jpg",
    image
)


print()
print(
    "Result image saved as: final_result.jpg"
)