import numpy as np
from sklearn.metrics import cohen_kappa_score

# Define the confusion matrix
confusion_matrix = np.array([
    [39, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 36, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 1, 37, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 39, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Flatten the confusion matrix to get the true and predicted labels
true_labels = []
predicted_labels = []

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        true_labels.extend([i] * confusion_matrix[i, j])
        predicted_labels.extend([j] * confusion_matrix[i, j])

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(true_labels, predicted_labels)
print(f"Cohen's Kappa: {kappa:.4f}")