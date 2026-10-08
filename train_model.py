import csv
from sklearn.neighbors import KNeighborsClassifier

# Store training data
X = []
y = []

# Read dataset
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

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

print("KNN model trained successfully!")

# Test the model
test_data = [[1, 0, 0]]

prediction = model.predict(test_data)

print("Test object classified as:", prediction[0])