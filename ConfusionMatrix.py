import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, balanced_accuracy_score
import numpy as np

# 1. Load the Excel file
file_path = r"IDMT_Categorized.xlsx"
df = pd.read_excel(file_path, header=0)  # First row is column headers

# 2. Print column names for debugging
print("Columns before renaming:", df.columns.tolist())

# 3. Drop unnecessary or unnamed columns (if they exist)
unnamed_cols = [col for col in df.columns if "Unnamed" in col]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)

# 4. Rename columns for clarity
df.rename(columns={
    'Class nr.': 'Predicted',  # Column E (predicted values)
    'Class': 'True'            # Column H (actual correct values)
}, inplace=True)

print("Columns after renaming:", df.columns.tolist())

# 5. Convert "Not Classified" to -1 and ensure numeric types
df['Predicted'] = df['Predicted'].fillna(-1)  # Replace NaN with -1
df['Predicted'] = df['Predicted'].apply(lambda x: -1 if x == "Not Classified" else int(x))
df['True'] = pd.to_numeric(df['True'], errors='coerce')  # Ensure numeric values

# Drop rows where 'True' contains NaN after conversion
df.dropna(subset=['True'], inplace=True)
df['True'] = df['True'].astype(int)  # Convert to integer after dropping NaNs

# 6. Filter data (keep only rows where True values are 0,1,2)
df = df[df['True'].isin([0, 1, 2])]
df = df[df['Predicted'].isin([-1, 0, 1, 2])]  # Allow -1 for "Not Classified"

# 7. Extract the true and predicted labels
y_true = df['True']
y_pred = df['Predicted']

# 8. Compute the confusion matrix including -1 for "Not Classified"
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, -1])
print("\nConfusion Matrix:\n", cm)

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Compute per-class metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# Print per-class metrics
class_labels = ["Car", "Bus/Truck", "Motorcycle"]
print("\nPer-Class Metrics:")
for i, label in enumerate(class_labels):
    print(f"{label}:")
    print(f"  - Precision: {precision[i]:.2%}")
    print(f"  - Recall: {recall[i]:.2%}")
    print(f"  - F1-score: {f1[i]:.2%}")

# Print balanced accuracy
print(f"\nBalanced Accuracy: {balanced_acc:.2%}")

# 9. Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels + ["Not Classified"], yticklabels=class_labels + ["Not Classified"])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix for Audio Classification')
plt.show()
