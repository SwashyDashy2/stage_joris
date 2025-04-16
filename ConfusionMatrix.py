import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report

# --- Configuration ---
excel_file = r"wavs\Blacklist results\car_1_beats_blf.xlsx"
# We'll load only columns B and E (zero-indexed: B = index 1, E = index 4)
df = pd.read_excel(excel_file, header=0, usecols=[4,5])

# For clarity, rename the columns:
# Assume column B header is "P Class" and column E header is "T Class"
df.rename(columns={"P Class": "Predicted", "T Class": "True"}, inplace=True)

# Debug: print the cleaned column names and a few rows
print("Data preview:")
print(df.head())

# Define allowed classes and final list of classes
ALLOWED_CLASSES = ["Car"]
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

# Create the confusion matrix using the FINAL_CLASSES order (counts)
cm = confusion_matrix(y_true, y_pred, labels=FINAL_CLASSES)

# Compute percentages per row (each row sums to 100%)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, None] * 100

# Create DataFrame for percentages
cm_df_pct = pd.DataFrame(cm_percentage, index=FINAL_CLASSES, columns=FINAL_CLASSES)

print("\nConfusion Matrix (Counts):")
print(pd.DataFrame(cm, index=FINAL_CLASSES, columns=FINAL_CLASSES))
print("\nConfusion Matrix (Percentages):")
print(cm_df_pct)

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

# Plot confusion matrix with percentages
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df_pct, annot=True, fmt='.1f', cmap='Blues')  # .1f shows one decimal
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix for Audio Classification')
plt.show()
