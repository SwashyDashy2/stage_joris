import json

root_name = "Human sounds"  # You can change this to "Human sounds" or any other name

def get_names_from_child_ids(data, root_name):
    # Find the entry with the given root_name
    def find_entry(name, data):
        for entry in data:
            if entry['name'] == name:
                return entry
        return None

    # Recursive function to gather names
    def gather_names(entry, data):
        names = [entry['name']]
        for child_id in entry['child_ids']:
            # Find the child entry by ID
            child_entry = next((item for item in data if item['id'] == child_id), None)
            if child_entry:
                names.extend(gather_names(child_entry, data))
        return names

    # Start the search with the root_name
    root_entry = find_entry(root_name, data)
    if root_entry:
        return gather_names(root_entry, data)
    else:
        return []

# Load data from the JSON file
with open("Depth_mapping_Mediapipe.json", "r") as file:
    data = json.load(file)

# Check if the root_name is a used name in the data file under the "name" tag
if any(entry['name'] == root_name for entry in data):
    # Example usage
    valid_names = get_names_from_child_ids(data, root_name)
else:
    raise ValueError(f"The root name '{root_name}' is not found in the data.")

most_likely_sources = ["Speech", "Human sounds", "Non-Valid Source", "Fart"]
time_stamps = [0.5, 1.0, 1.5, 2.0]

# Compare and filter out invalid sources and their corresponding time stamps
filtered_sources = []
filtered_time_stamps = []

for source, timestamp in zip(most_likely_sources, time_stamps):
    if source in valid_names:
        filtered_sources.append(source)
        filtered_time_stamps.append(timestamp)

# Now filtered_sources and filtered_time_stamps only contain valid entries
print("Filtered Sources:", filtered_sources)
print("Filtered Time Stamps:", filtered_time_stamps)
