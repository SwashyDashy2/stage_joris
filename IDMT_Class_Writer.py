import pandas as pd

# Define the categorization function
def categorize_vehicle(filename):
    # Car
    if "_CR_" in filename or "_CL_" in filename:
        return 0
    # Bus or Truck
    elif any(x in filename for x in ["_BR_", "_BL_", "_TR_", "_TL_"]):
        return 1
    # Motorcycle
    elif "_MR_" in filename or "_ML_" in filename:
        return 2
    # No match
    else:
        return None

# Load the Excel file
file_path = r"IDMT.xlsx"
df = pd.read_excel(file_path)

# Apply categorization to filenames in column A
df["Class"] = df.iloc[:, 0].apply(categorize_vehicle)

# Save the updated file
output_path = r"IDMT_Categorized.xlsx"
df.to_excel(output_path, index=False)

print(f"Classification complete! Results saved to {output_path}")
