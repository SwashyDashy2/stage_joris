import os
import shutil

# Define the path of the parent folder containing all the subfolders
parent_folder = r'wavs\ESC10'

# List all the subfolders in the parent folder
subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# Loop through each subfolder and move the files to the parent folder
for subfolder in subfolders:
    # List all files in the subfolder
    files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
    
    for file in files:
        # Define the full paths for the current file and the destination (parent folder)
        old_path = os.path.join(subfolder, file)
        new_path = os.path.join(parent_folder, file)
        
        # Move the file to the parent folder (will overwrite if the file with the same name exists)
        shutil.move(old_path, new_path)
        print(f"Moved {file} to {parent_folder}")
    
    # After moving all files, remove the subfolder
    os.rmdir(subfolder)
    print(f"Removed folder {subfolder}")
