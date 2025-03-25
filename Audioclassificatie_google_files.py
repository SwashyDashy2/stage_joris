import numpy as np
import os
import json
from scipy.io import wavfile
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter, defaultdict
from tqdm import tqdm

# --- Configuration Parameters ---
AUDIO_FILE_PATH = r'wavs\ESC10'
ANALYZE_FILE = False  # If True, an Excel summary will be created for each audio file.
EXCEL_OUTPUT_NAME = "ESC10_f_avg.xlsx"

# --- Audio Classifier Parameters ---
N_CLASSES = 5
TOP_CLASSIFICATIONS = 5 # Number of top classifications to use for the conclusion.

# --- Filter Options ---
APPLY_ROOT_CLASS_FILTER = False  # When True, only classes (and their children) that are in valid_names are registered.
ROOT_NAME = "Motor vehicle (road)"  # Example root category.

# New variables for selective filtering:
ALLOWED_CLASSES = ["Baby cry, infant cry", "Baby laughter", "Chainsaw", "Chicken, rooster", "Helicopter", "Fire", "Dog", "Rain", "Waves, surf", "Tick", "Tick-tock", "Sneeze"]
SELECTIVE_FILTER = False  # If True, only display classes (in the final top 5) that are part of ALLOWED_CLASSES

if APPLY_ROOT_CLASS_FILTER and SELECTIVE_FILTER:
    raise ValueError("APPLY_ROOT_CLASS_FILTER and SELECTIVE_FILTER cannot both be True at the same time.")

# --- Initialize Paths ---
MODEL_PATH = 'yamnet_classifier.tflite'
JSON_PATH = "Depth_mapping_Mediapipe.json"
INTERVAL_MS = 975  # Analysis interval in milliseconds

# --- Helper Functions ---

def classify_audio(audio_file_path, model_path, n_classes, interval_ms):
    """
    Loads the audio and classifier model, performs classification, and processes the results.
    Returns a dictionary with:
      - sample_rate, audio_data, classification_results, duration,
      - all_top10: a list of lists with tuples (class, probability) per time segment,
      - class_counts: a counter of the classifications.
    """
    sample_rate, audio_data = wavfile.read(audio_file_path)
    audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=n_classes)
    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = containers.AudioData.create_from_array(
            audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate
        )
        classification_results = classifier.classify(audio_clip)
    
    duration = len(audio_data) / sample_rate

    all_top10 = []
    class_counts = Counter()
    
    for timestamp in range(0, int(duration * 1000), interval_ms):
        idx = timestamp // interval_ms
        if idx < len(classification_results):
            classifications = classification_results[idx].classifications[0].categories
            top10 = [(cat.category_name, cat.score) for cat in classifications[:n_classes]]
            all_top10.append(top10)
            
            print(f"\nTimestamp {timestamp / 1000:.3f} seconds in file '{os.path.basename(audio_file_path)}':")
            for rank, (cls, prob) in enumerate(top10, 1):
                print(f"  {rank}. {cls}: {prob:.2f}")
            
            for category in classifications[:n_classes]:
                cls_name = category.category_name
                if APPLY_ROOT_CLASS_FILTER:
                    if cls_name in valid_names:
                        class_counts[cls_name] += 1
                elif SELECTIVE_FILTER:
                    if cls_name in ALLOWED_CLASSES:
                        class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] += 1

    return {
        'sample_rate': sample_rate,
        'audio_data': audio_data,
        'classification_results': classification_results,
        'duration': duration,
        'all_top10': all_top10,
        'class_counts': dict(class_counts)
    }

def get_valid_names(data, root_name):
    """Recursively retrieve all valid names from the JSON data, starting from the specified root category."""
    def find_entry(name):
        return next((entry for entry in data if entry['name'] == name), None)
    
    def gather_names(entry):
        names = [entry['name']]
        for child_id in entry['child_ids']:
            child_entry = next((item for item in data if item['id'] == child_id), None)
            if child_entry:
                names.extend(gather_names(child_entry))
        return names
    
    root_entry = find_entry(root_name)
    return gather_names(root_entry) if root_entry else []

def build_json_mapping(json_data):
    """Build a mapping: class name -> JSON entry."""
    mapping = {}
    for entry in json_data:
        mapping[entry['name']] = entry
    return mapping

def is_descendant(child_name, parent_name, json_mapping):
    """
    Recursively check if child_name is a descendant of parent_name via the JSON structure.
    If parent_name is in json_mapping, check its child_ids and look for child_name.
    """
    if parent_name not in json_mapping:
        return False
    parent_entry = json_mapping[parent_name]
    if not parent_entry.get('child_ids'):
        return False
    for child_id in parent_entry['child_ids']:
        child_entry = next((item for item in json_data if item.get('id') == child_id), None)
        if child_entry:
            if child_entry['name'] == child_name:
                return True
            if is_descendant(child_name, child_entry['name'], json_mapping):
                return True
    return False

# --- Load JSON and build Depth Mapping ---
with open(JSON_PATH, "r") as file:
    json_data = json.load(file)

if not any(entry['name'] == ROOT_NAME for entry in json_data):
    raise ValueError(f"Root name '{ROOT_NAME}' not found in JSON data.")

valid_names = get_valid_names(json_data, ROOT_NAME)
depth_map = {entry['name']: entry.get('depth', "N/A") for entry in json_data}
json_mapping = build_json_mapping(json_data)

# --- Main Processing Block ---
if os.path.isdir(AUDIO_FILE_PATH):
    wav_files = [os.path.join(AUDIO_FILE_PATH, f) for f in os.listdir(AUDIO_FILE_PATH) if f.lower().endswith('.wav')]
    
    summary_data = []
    total_files = len(wav_files)
    for idx, file_path in enumerate(wav_files, start=1):
        print(f"\n============================================")
        print(f"Processing file {idx}/{total_files}: {os.path.basename(file_path)}")
        progress = (idx / total_files) * 100
        print(f"Progress: [{int(progress)}%] {'█' * (idx * 20 // total_files)}")
    
        results = classify_audio(file_path, MODEL_PATH, N_CLASSES, INTERVAL_MS)
        
        # Accumulate probabilities per class from the time segments.
        class_probabilities = defaultdict(list)
        for timestamp_results in results['all_top10']:
            for cls, prob in timestamp_results:
                if APPLY_ROOT_CLASS_FILTER:
                    if cls in valid_names:
                        class_probabilities[cls].append(prob)
                elif SELECTIVE_FILTER:
                    if cls in ALLOWED_CLASSES:
                        class_probabilities[cls].append(prob)
                else:
                    class_probabilities[cls].append(prob)
        
        # Calculate the average probability per class.
        avg_probabilities = {}
        for cls, prob_list in class_probabilities.items():
            if prob_list:
                avg_probabilities[cls] = sum(prob_list) / len(prob_list)
        
        # Filter on ALLOWED_CLASSES if SELECTIVE_FILTER is active.
        if SELECTIVE_FILTER:
            avg_probabilities = {cls: avg for cls, avg in avg_probabilities.items() if cls in ALLOWED_CLASSES}
        
        print("\nTop 5 classifications (average probability with depth information):")
        for cls, avg in sorted(avg_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:
            depth = depth_map.get(cls, "N/A")
            print(f"  {cls}: {avg:.2f} (Depth = {depth})")
        
        # Determine the top 5 based on average probability.
        top5 = sorted(avg_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
        # Build a list of candidates: (class, average probability, depth)
        candidates = []
        for cls, avg in top5:
            depth_val = depth_map.get(cls, None)
            try:
                depth_numeric = float(depth_val)
            except (TypeError, ValueError):
                depth_numeric = 0
            candidates.append((cls, avg, depth_numeric))
        
        # Start with the candidate with the highest average probability.
        if candidates:
            best_candidate = candidates[0]
            best_cls, best_avg, best_depth = best_candidate
            # Loop over the remaining candidates: if they are a descendant of the current candidate and have a greater depth, update the candidate.
            for cand in candidates[1:]:
                cand_cls, cand_avg, cand_depth = cand
                if is_descendant(cand_cls, best_cls, json_mapping) and cand_depth > best_depth:
                    best_candidate = cand
                    best_cls, best_avg, best_depth = cand
            conclusion_str = (f"Conclusion: The sound is likely a {best_cls}. "
                              f"Depth = {int(best_depth)} and average probability = {best_avg:.2f}.")
        else:
            best_cls, best_avg, best_depth = "N/A", "N/A", "N/A"
            conclusion_str = "Conclusion: No valid depth information available."
        
        print(conclusion_str)
        
        summary_data.append({
            "File": os.path.basename(file_path),
            "Most Likely Sound": best_cls,
            "Depth": best_depth,
            "Average Probability": best_avg
        })
    
    if ANALYZE_FILE:
        import pandas as pd
        df_summary = pd.DataFrame(summary_data)
        excel_output_path = os.path.join(AUDIO_FILE_PATH, EXCEL_OUTPUT_NAME)
        df_summary.to_excel(excel_output_path, index=False)
        print(f"\nExcel file created: {excel_output_path}")
    else:
        print("\nSummary Data:")
        for row in summary_data:
            print(row)
else:
    print(f"Processing single file: {AUDIO_FILE_PATH}")
    results = classify_audio(AUDIO_FILE_PATH, MODEL_PATH, N_CLASSES, INTERVAL_MS)
    
    class_probabilities = defaultdict(list)
    for timestamp_results in results['all_top10']:
        for cls, prob in timestamp_results:
            if APPLY_ROOT_CLASS_FILTER:
                if cls in valid_names:
                    class_probabilities[cls].append(prob)
            elif SELECTIVE_FILTER:
                if cls in ALLOWED_CLASSES:
                    class_probabilities[cls].append(prob)
            else:
                class_probabilities[cls].append(prob)
    
    avg_probabilities = {}
    for cls, prob_list in class_probabilities.items():
        if prob_list:
            avg_probabilities[cls] = sum(prob_list) / len(prob_list)
    
    if SELECTIVE_FILTER:
        avg_probabilities = {cls: avg for cls, avg in avg_probabilities.items() if cls in ALLOWED_CLASSES}
    print("\nTop 5 classifications (average probability with depth information):")
    for cls, avg in sorted(avg_probabilities.items(), key=lambda item: item[1], reverse=True)[:TOP_CLASSIFICATIONS]:
        depth = depth_map.get(cls, "N/A")
        print(f"  {cls}: {avg:.2f} (Depth = {depth})")
    
    top5 = sorted(avg_probabilities.items(), key=lambda item: item[1], reverse=True)[:TOP_CLASSIFICATIONS]
    candidates = []
    for cls, avg in top5:
        depth_val = depth_map.get(cls, None)
        try:
            depth_numeric = float(depth_val)
        except (TypeError, ValueError):
            depth_numeric = 0
        candidates.append((cls, avg, depth_numeric))
    if candidates:
        best_candidate = candidates[0]
        best_cls, best_avg, best_depth = best_candidate
        for cand in candidates[1:]:
            cand_cls, cand_avg, cand_depth = cand
            if is_descendant(cand_cls, best_cls, json_mapping) and cand_depth > best_depth:
                best_candidate = cand
                best_cls, best_avg, best_depth = cand
        print(f"\nConclusion: The sound is likely a {best_cls}. "
              f"Depth = {int(best_depth)} and average probability = {best_avg:.2f}.")
    else:
        print("\nConclusion: No valid depth information available in the top classifications.")
