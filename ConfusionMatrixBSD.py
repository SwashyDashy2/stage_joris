import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, classification_report

# --- Configuration ---
excel_file = r"BSD_f.xlsx"
# We'll load only columns B and E (zero-indexed: B = index 1, E = index 4)
df = pd.read_excel(excel_file, header=0, usecols=[1,4])

# For clarity, rename the columns:
# Assume column B header is "Predicted Class" and column E header is "Correct Class"
df.rename(columns={"P Class": "Predicted", "T Class": "True"}, inplace=True)

# Debug: print the cleaned column names and a few rows
print("Data preview:")
print(df.head())

# Define allowed classes and final list of classes
ALLOWED_CLASSES = ["Car", "Motorcycle",  "Truck", "Honk", "Emergency vehicle",]
FINAL_CLASSES = ALLOWED_CLASSES + ["Not Classified"]

print("Unique values in True column:", df["True"].unique())
print("Unique values in Predicted column:", df["Predicted"].unique())


# Convert both columns to string and strip whitespace
df["Predicted"] = df["Predicted"].astype(str).str.strip()
df["True"] = df["True"].astype(str).str.strip()

# Map any predicted (and true) value not in ALLOWED_CLASSES to "Not Classified"
df["Predicted"] = df["Predicted"].apply(lambda x: x if x in ALLOWED_CLASSES else "Not Classified")
df["True"] = df["True"].apply(lambda x: x if x in ALLOWED_CLASSES else "Not Classified")

# Extract labels for confusion matrix
y_pred = df["Predicted"]
y_true = df["True"]

# Create the confusion matrix using the FINAL_CLASSES order
cm = confusion_matrix(y_true, y_pred, labels=FINAL_CLASSES)
cm_df = pd.DataFrame(cm, index=FINAL_CLASSES, columns=FINAL_CLASSES)

print("\nConfusion Matrix:")
print(cm_df)

# Compute overall metrics
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")
print(f"Balanced Accuracy: {balanced_acc:.2%}")

# Compute per-class metrics using classification_report for more detail
report_dict = classification_report(y_true, y_pred, labels=FINAL_CLASSES, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("\nClassification Report:")
print(report_df[['precision', 'recall', 'f1-score', 'support']])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix for Audio Classification')
plt.show()
