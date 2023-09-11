import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models

# Load the dataset
cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
df = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset.arff', names=cols)
classes = pd.get_dummies(df['Class'])
df = pd.concat([df, classes], axis=1)
df.drop(columns=['Class'], inplace=True)

# Splitting the data into train, test and validation sets
X = df.iloc[:, :-7].values
y = df.iloc[:, -7:].values

# First, split into 80% training and 20% temp (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the temp data into 50% test and 50% validation
X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
X_validation_normalized = scaler.transform(X_validation)

# Define the neural network model using TensorFlow
model = models.Sequential([
    layers.Dense(64, activation='sigmoid', input_shape=(X_train_normalized.shape[1],)),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 100
history = model.fit(X_train_normalized, y_train, epochs=epochs, validation_data=(X_validation_normalized, y_validation), verbose=0)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test)

# Extract final loss, accuracy and val_accuracy values
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

# Plot training errors
plt.figure(figsize=(7, 5))
plt.plot(range(epochs), history.history['loss'])
plt.annotate(f'Final Loss: {final_loss:.4f}', xy=(epochs - 1, final_loss), xytext=(epochs - 10, final_loss + 0.05), arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Plot accuracies
plt.figure(figsize=(11, 5))
plt.plot(range(epochs), history.history['accuracy'], label='Train Accuracy')
plt.plot(range(epochs), history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(range(epochs), [test_accuracy] * epochs, label='Test Accuracy', linestyle='--')
plt.annotate(f'Final Train Accuracy: {final_accuracy:.4f}', xy=(epochs - 1, final_accuracy), xytext=(epochs - 16, final_accuracy + 0.0), arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Final Validation Accuracy: {final_val_accuracy:.4f}', xy=(epochs - 1, final_val_accuracy), xytext=(epochs - 10, final_val_accuracy + 0.02), arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Test Accuracy: {test_accuracy:.4f}', xy=(epochs - 1, test_accuracy), xytext=(epochs - 13, test_accuracy + 0.05), arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# Calculate the confusion matrix
y_pred = model.predict(X_test_normalized)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix as a heatmap
labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate key metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Print the results
print("Test Accuracy:", test_accuracy)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)