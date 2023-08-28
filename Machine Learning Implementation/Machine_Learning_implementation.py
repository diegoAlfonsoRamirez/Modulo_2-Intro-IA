import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hypothesis(theta_0, theta_1, x):
    return theta_0 + np.dot(x, theta_1)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 0.0001) + (1 - y_true) * np.log(1.0001 - y_pred))

# Load the dataset and perform one-hot encoding
cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']
df = pd.read_csv('DryBeanDataset/Dry_Bean_Dataset.arff', names=cols)
classes = pd.get_dummies(df['Class'])
df = pd.concat([df, classes], axis=1)
df.drop(columns=['Class'], inplace=True)

# Splitting the data into train and test sets
np.random.seed(42)
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train_size = int(0.8 * len(shuffled_df))
train_data = shuffled_df.iloc[:train_size]
test_data = shuffled_df.iloc[train_size:]

# Extract features and labels
X_train = train_data.iloc[:, :-7].values
y_train = train_data.iloc[:, -7:].values
X_test = test_data.iloc[:, :-7].values
y_test = test_data.iloc[:, -7:].values

# Normalize the data tran and test
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train_normalized = (X_train - X_train_mean) / X_train_std
X_test_normalized = (X_test - X_train_mean) / X_train_std

# Initialize parameters and hyperparameters
theta_0 = np.zeros(7)
theta_1 = np.zeros((16, 7))
learning_rate = 0.05
epochs = 300

# Lists to store errors and accuracies during training
train_errors = []
train_accuracies = []
test_accuracies = []

# Training loop
for epoch in range(epochs):
    error = 0
    correct_train = 0
    
    for i in range(len(X_train_normalized)):
        x_i = X_train_normalized[i]
        y_i = y_train[i]
        
        z = np.dot(x_i, theta_1) + theta_0
        h = sigmoid(z)
        
        error += cross_entropy(y_i, h)
        
        predicted_class = np.argmax(h)
        true_class = np.argmax(y_i)
        
        if predicted_class == true_class:
            correct_train += 1
        
        delta_theta_0 = (h - y_i).sum()
        delta_theta_1 = np.outer(x_i, h - y_i)
        
        theta_0 -= (learning_rate / len(X_train)) * delta_theta_0
        theta_1 -= (learning_rate / len(X_train)) * delta_theta_1
    
    train_errors.append(error)
    train_accuracy = correct_train / len(X_train_normalized)
    train_accuracies.append(train_accuracy)
    
    # Calculate test accuracy
    correct_test = 0
    for i in range(len(X_test_normalized)):
        x_i = X_test_normalized[i]
        y_i = y_test[i]
        
        z = np.dot(x_i, theta_1) + theta_0
        h = sigmoid(z)
        
        predicted_class = np.argmax(h)
        true_class = np.argmax(y_i)
        
        if predicted_class == true_class:
            correct_test += 1
    
    test_accuracy = correct_test / len(X_test_normalized)
    test_accuracies.append(test_accuracy)

# Confusion matrix
confusion_matrix = np.zeros((7, 7))

for i in range(len(X_test_normalized)):
    x_i = X_test_normalized[i]
    y_i = y_test[i]
    
    z = np.dot(x_i, theta_1) + theta_0
    h = sigmoid(z)
    
    predicted_class = np.argmax(h)
    true_class = np.argmax(y_i)
    
    confusion_matrix[true_class, predicted_class] += 1

print('Confusion Matrix:')
print(confusion_matrix)
'''
# Plot training errors
plt.plot(range(epochs), train_errors)
plt.xlabel('Epoch')
plt.ylabel('Training Error (Cross-Entropy)')
plt.title('Training Error Over Epochs')
'''
# Plot accuracies
plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(epochs), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()