import pandas as pd

# 1. Define the categorization function
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

# 2. Load the Excel file
file_path = r"IDMT_Traffic\audio\IDMT_data_Classified_ava_prob.xlsx"
df = pd.read_excel(file_path)

# 3. Apply the function to the "File" column
df["Category"] = df["File"].apply(categorize_vehicle)

# 4. (Optional) Rename "Category" to "E" if you specifically want the column named "E"
# df.rename(columns={"Category": "E"}, inplace=True)

# 5. Save the results to a new Excel file
output_path = "categorized_output_ava_prob.xlsx"
df.to_excel(output_path, index=False)
print(f"Classification complete! Results saved to {output_path}")
