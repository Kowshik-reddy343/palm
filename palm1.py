from google.colab import drive
drive.mount('/content/drive')
import os

from google.colab import drive
drive.mount('/content/drive')

# Access the folder
folder_path = '/content/drive/MyDrive/Palm_Processed'

# Check if the folder exists
if os.path.exists(folder_path):
  print(f"Successfully accessed folder: {folder_path}")

  # List the files in the folder (optional)
  for filename in os.listdir(folder_path):
    print(filename)
else:
    print(f"Error: Folder not found at {folder_path}")

    image_count = 0
    # Count image files (you might need to adjust the extensions)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add more extensions if needed
            image_count += 1
    print(f"Total image files found: {image_count}")

    if os.path.exists(folder_path):
        print(f"Successfully accessed folder: {folder_path}")

        labels = []
        filenames = []

        # List the files in the folder and assign labels
        for filename in os.listdir(folder_path):
            if filename.startswith("Anemic"):
                labels.append(1)
            elif filename.startswith("Non-Anemic"):
                labels.append(0)
            else:
                print(f"Warning: Skipping file '{filename}' due to unexpected naming convention.")
                continue  # Skip files that don't match the expected naming

            filenames.append(filename)

        # Now you have two lists: 'filenames' and 'labels'
        # You can use these lists for further processing, like creating a dataframe
        # Example:
        import pandas as pd
        df = pd.DataFrame({'filename': filenames, 'label': labels})
        print(df.head())

label_counts = df['label'].value_counts()
print("\nNumber of images for each label:")
print(label_counts)

unlabeled_count = len([filename for filename in os.listdir(folder_path) if not filename.startswith(("Anemic", "Non-Anemic"))])
print(f"\nNumber of images not labeled as 0 or 1: {unlabeled_count}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from PIL import Image


# Assuming 'df' is the DataFrame created in the previous code snippet
# ... (your code to create 'df' as shown in the original prompt) ...

# Feature Extraction (example: using pixel values as features - replace with actual feature extraction)
def extract_features(image_path):
    try:
      img = Image.open(image_path).convert("L") # Convert to grayscale
      img = img.resize((64,64)) # Resize for consistency
      img_array = np.array(img)
      return img_array.flatten()
    except Exception as e:
      print(f"Error processing {image_path}: {e}")
      return None


folder_path = '/content/drive/MyDrive/Palm_Processed'
features = []
labels = []

for index, row in df.iterrows():
    image_path = os.path.join(folder_path, row['filename'])
    extracted_features = extract_features(image_path)
    if extracted_features is not None:
        features.append(extracted_features)
        labels.append(row['label'])


# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # You can adjust parameters
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
import numpy as np # Import numpy

# Convert X_train and X_test to NumPy arrays if they are EagerTensors
X_train = X_train.numpy()
X_test = X_test.numpy()

# Reshape X_train and X_test to 2D
X_train = X_train.reshape(X_train.shape[0], -1)  # Reshape to (num_samples, num_features)
X_test = X_test.reshape(X_test.shape[0], -1)    # Reshape to (num_samples, num_features)

# Initialize and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5) # You can adjust the number of neighbors
knn_classifier.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_classifier.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn}")
print(classification_report(y_test, y_pred_knn))

from sklearn.naive_bayes import GaussianNB

# Initialize and train the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

# Predictions
y_pred_gnb = gnb_classifier.predict(X_test)

# Evaluate the model
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"Naive Bayes Accuracy: {accuracy_gnb}")
print(classification_report(y_test, y_pred_gnb))


from sklearn.tree import DecisionTreeClassifier

# ... (Your existing code for data loading and feature extraction) ...

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)  # You can adjust hyperparameters
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")
print(classification_report(y_test, y_pred_dt))

# !pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Feature Extraction (example: using pixel values as features)
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img = img.resize((64, 64))  # Resize for consistency
        img_array = np.array(img)
        return img_array.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Assuming you have already created 'df' DataFrame in previous steps
# ... (Your code to create 'df' with 'filename' and 'label' columns) ...

folder_path = '/content/drive/MyDrive/Palm_Processed'
features = []
labels = []

for index, row in df.iterrows():
    image_path = os.path.join(folder_path, row['filename'])
    extracted_features = extract_features(image_path)
    if extracted_features is not None:
        features.append(extracted_features)
        labels.append(row['label'])

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Ensure X and y have the same number of samples before reshaping
X = X[:len(y)]  # Adjust X to match the length of y

# Split data (do this after adjusting X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for 3D CNN
img_height, img_width = 64, 64
X_train = X_train.reshape(-1, img_height, img_width, 1)  # Add a channel dimension for grayscale
X_test = X_test.reshape(-1, img_height, img_width, 1)

# 3D CNN Model
model = keras.Sequential(
    [
        keras.Input(shape=(img_height, img_width, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(10, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # Output layer for binary classification
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ... (Your existing code for data loading, preprocessing) ...

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,      # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2, # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,       # Randomly apply shear transformations
    zoom_range=0.2,        # Randomly zoom in/out on images
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill any newly created pixels with the nearest pixel value
)

# Define the optimal probability-based DNN model with improved architecture
def create_probability_dnn(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),  # Input layer
        layers.Flatten(),  # Flatten the input if it's not already 1D
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # Increased neurons, L2 regularization
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # Increased neurons, L2 regularization
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid for probability
    ])
    return model

# Assuming you have already created X_train, y_train, X_test, y_test
# ... (Your code to create 'df' with 'filename' and 'label' columns) ...

folder_path = '/content/drive/MyDrive/Palm_Processed'
features = []
labels = []

for index, row in df.iterrows():
    image_path = os.path.join(folder_path, row['filename'])
    extracted_features = extract_features(image_path)
    if extracted_features is not None:
        features.append(extracted_features)
        labels.append(row['label'])

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Reshape for DNN
X = X.reshape(X.shape[0], -1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]

# Create and compile the model
model = create_probability_dnn(input_shape)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with data augmentation and early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=100,  # Increased epochs
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Get probability predictions
probability_predictions = model.predict(X_test)

# Example: Print the predicted probabilities
print("Probability Predictions:")
print(probability_predictions[:5])





