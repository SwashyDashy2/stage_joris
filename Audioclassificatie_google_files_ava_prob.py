import numpy as np
import os
import json
from scipy.io import wavfile
from IPython.display import Audio, display
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter, defaultdict
from tqdm import tqdm

# --- Configuration Parameters ---
AUDIO_FILE_PATH = r'IDMT_Traffic\audio'
ANALYZE_FILE = True  # Indien True, wordt voor elk audiobestand een Excel-samenvatting aangemaakt.
excel_output_name = "IDMT_data_Classified_ava_prob.xlsx"

# --- Audio Classifier Parameters ---
N_CLASSES = 10

# --- Filter Options ---
APPLY_ROOT_CLASS_FILTER = False  # When True, only classes (and their children) that are in valid_names are registered.
ROOT_NAME = "Motor vehicle (road)"  # Voorbeeld rootcategorie.

# New variables for selective filtering:
ALLOWED_CLASSES = ["Car", "Motorcycle", "Bus", "Truck"]
SELECTIVE_FILTER = True  # If True, only display classes (in the final top 5) that are part of ALLOWED_CLASSES

# Ensure that both filters are not enabled simultaneously.
if APPLY_ROOT_CLASS_FILTER and SELECTIVE_FILTER:
    raise ValueError("APPLY_ROOT_CLASS_FILTER and SELECTIVE_FILTER cannot both be True at the same time.")

# --- New variable for maximum allowed depth in conclusion ---
# Set MAX_DEPTH to None to remove the restriction, or set it to a number (e.g., 3) to limit the maximum depth.
MAX_DEPTH = 3

# --- Initialize Paths ---
MODEL_PATH = 'yamnet_classifier.tflite'
JSON_PATH = "Depth_mapping_Mediapipe.json"
INTERVAL_MS = 975  # Analyse-interval in milliseconden

# --- Function Definitions ---

def classify_audio(audio_file_path, model_path, n_classes, interval_ms):
    """
    Laadt de audio en het classifiermodel, voert classificatie uit en verwerkt de resultaten.
    
    Retourneert een dictionary met:
      - sample_rate, audio_data, classification_results, duration,
      - all_top10: een lijst van lijsten met tuples (class, probability) per tijdssegment,
      - class_counts: een teller van de classificaties, eventueel gefilterd op valid_names of ALLOWED_CLASSES.
    """
    # Laad en normaliseer audio
    sample_rate, audio_data = wavfile.read(audio_file_path)
    audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
    
    # Laad model en voer classificatie uit
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=n_classes)
    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = containers.AudioData.create_from_array(
            audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate
        )
        classification_results = classifier.classify(audio_clip)
    
    # Bepaal de duur van de audio (in seconden)
    duration = len(audio_data) / sample_rate

    all_top10 = []  # Lijst van lijsten: elk element bevat tuples (class, probability) per tijdssegment.
    class_counts = Counter()
    
    # Verwerk de classificatieresultaten per vast tijdsinterval.
    for timestamp in range(0, int(duration * 1000), interval_ms):
        idx = timestamp // interval_ms
        if idx < len(classification_results):
            classifications = classification_results[idx].classifications[0].categories
            # Bouw een lijst met (class, probability) voor de top predictions
            top10 = [(cat.category_name, cat.score) for cat in classifications[:n_classes]]
            all_top10.append(top10)
            
            print(f"\nTimestamp {timestamp / 1000:.3f} seconds in file '{os.path.basename(audio_file_path)}':")
            for rank, (cls, prob) in enumerate(top10, 1):
                print(f"  {rank}. {cls}: {prob:.2f}")
            
            # Tel alle classificaties, met filtering afhankelijk van de ingestelde opties.
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
    """Haal recursief alle geldige namen op uit de JSON-data, beginnend vanaf de opgegeven rootcategorie."""
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

# --- Laad JSON en bouw Depth Mapping ---
with open(JSON_PATH, "r") as file:
    json_data = json.load(file)

if not any(entry['name'] == ROOT_NAME for entry in json_data):
    raise ValueError(f"Root name '{ROOT_NAME}' not found in JSON data.")

valid_names = get_valid_names(json_data, ROOT_NAME)
depth_map = {entry['name']: entry.get('depth', "N/A") for entry in json_data}

# --- Hoofdverwerkingsblok ---

if os.path.isdir(AUDIO_FILE_PATH):
    wav_files = [os.path.join(AUDIO_FILE_PATH, f) for f in os.listdir(AUDIO_FILE_PATH) if f.lower().endswith('.wav')]
    
    summary_data = []  # Houdt de samenvattingsinformatie per bestand bij.
    
    total_files = len(wav_files)
    for idx, file_path in enumerate(wav_files, start=1):
        print(f"\n============================================")
        print(f"Processing file {idx}/{total_files}: {os.path.basename(file_path)}")
    
        # Optional: Visual progress bar
        progress = (idx / total_files) * 100
        print(f"Progress: [{int(progress)}%] {'█' * (idx * 20 // total_files)}")
    
        results = classify_audio(file_path, MODEL_PATH, N_CLASSES, INTERVAL_MS)
        
        # Tel de totale kansen per class over alle tijdssegmenten, met filtering afhankelijk van de ingestelde opties.
        total_probabilities = defaultdict(float)
        for timestamp_results in results['all_top10']:
            for cls, prob in timestamp_results:
                if APPLY_ROOT_CLASS_FILTER:
                    if cls in valid_names:
                        total_probabilities[cls] += prob
                elif SELECTIVE_FILTER:
                    if cls in ALLOWED_CLASSES:
                        total_probabilities[cls] += prob
                else:
                    total_probabilities[cls] += prob

        # Selecteer de top 5 classes op basis van de totale kansen.
        top5 = sorted(total_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
        # If SELECTIVE_FILTER is true, further filter the top5 list to only include allowed classes.
        if SELECTIVE_FILTER:
            top5 = [(cls, total) for cls, total in top5 if cls in ALLOWED_CLASSES]
            
        print("\nTop 5 classificaties (met diepte-informatie):")
        for cls, total in top5:
            depth = depth_map.get(cls, "N/A")
            print(f"  {cls}: {total:.2f} (Diepte = {depth})")

        # Bepaal de meest waarschijnlijke geluidscategorie op basis van de hoogste diepte en kans.
        top5_with_depth = []
        for cls, total in top5:
            depth = depth_map.get(cls, None)
            try:
                depth_numeric = float(depth)
            except (TypeError, ValueError):
                depth_numeric = None
            if depth_numeric is not None:
                top5_with_depth.append((cls, total, depth_numeric))
        
        if top5_with_depth:
            # Filter candidates based on MAX_DEPTH if it is set.
            if MAX_DEPTH is not None:
                filtered_candidates = [x for x in top5_with_depth if x[2] <= MAX_DEPTH]
                # If there are valid candidates within the MAX_DEPTH limit, use them.
                if filtered_candidates:
                    max_depth = max(x[2] for x in filtered_candidates)
                    candidates = [x for x in filtered_candidates if x[2] == max_depth]
                else:
                    # If none are within the MAX_DEPTH limit, fall back to using all candidates.
                    max_depth = max(x[2] for x in top5_with_depth)
                    candidates = [x for x in top5_with_depth if x[2] == max_depth]
            else:
                max_depth = max(x[2] for x in top5_with_depth)
                candidates = [x for x in top5_with_depth if x[2] == max_depth]
            best_candidate = max(candidates, key=lambda x: x[1])
            best_cls, best_total, best_depth = best_candidate
            conclusion_str = (f"Conclusie: Het geluid is waarschijnlijk een {best_cls}. "
                              f"Hoogste diepte = {int(best_depth)} en totale waarschijnlijkheid = {best_total:.2f}.")
        else:
            best_cls, best_total, best_depth = "N/A", "N/A", "N/A"
            conclusion_str = "Conclusie: Geen geldige diepte-informatie beschikbaar."
        
        print(conclusion_str)
        
        summary_data.append({
            "File": os.path.basename(file_path),
            "Most Likely Sound": best_cls,
            "Depth": best_depth,
            "Summated Probability": best_total
        })
    
    if ANALYZE_FILE:
        import pandas as pd
        df_summary = pd.DataFrame(summary_data)
        excel_output_path = os.path.join(AUDIO_FILE_PATH, excel_output_name)
        df_summary.to_excel(excel_output_path, index=False)
        print(f"\nExcel file created: {excel_output_path}")
    else:
        print("\nSummary Data:")
        for row in summary_data:
            print(row)
else:
    print(f"Processing single file: {AUDIO_FILE_PATH}")
    results = classify_audio(AUDIO_FILE_PATH, MODEL_PATH, N_CLASSES, INTERVAL_MS)
    
    total_probabilities = defaultdict(float)
    for timestamp_results in results['all_top10']:
        for cls, prob in timestamp_results:
            if APPLY_ROOT_CLASS_FILTER:
                if cls in valid_names:
                    total_probabilities[cls] += prob
            elif SELECTIVE_FILTER:
                if cls in ALLOWED_CLASSES:
                    total_probabilities[cls] += prob
            else:
                total_probabilities[cls] += prob
    top5 = sorted(total_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    if SELECTIVE_FILTER:
        top5 = [(cls, total) for cls, total in top5 if cls in ALLOWED_CLASSES]
    print("\nTop 5 classificaties (met diepte-informatie):")
    for cls, total in top5:
        depth = depth_map.get(cls, "N/A")
        print(f"  {cls}: {total:.2f} (Diepte = {depth})")
    
    top5_with_depth = []
    for cls, total in top5:
        depth = depth_map.get(cls, None)
        try:
            depth_numeric = float(depth)
        except (TypeError, ValueError):
            depth_numeric = None
        if depth_numeric is not None:
            top5_with_depth.append((cls, total, depth_numeric))
    if top5_with_depth:
        if MAX_DEPTH is not None:
            filtered_candidates = [x for x in top5_with_depth if x[2] <= MAX_DEPTH]
            if filtered_candidates:
                max_depth = max(x[2] for x in filtered_candidates)
                candidates = [x for x in filtered_candidates if x[2] == max_depth]
            else:
                max_depth = max(x[2] for x in top5_with_depth)
                candidates = [x for x in top5_with_depth if x[2] == max_depth]
        else:
            max_depth = max(x[2] for x in top5_with_depth)
            candidates = [x for x in top5_with_depth if x[2] == max_depth]
        best_candidate = max(candidates, key=lambda x: x[1])
        best_cls, best_total, best_depth = best_candidate
        print(f"\nConclusie: Het geluid is waarschijnlijk een {best_cls}. "
              f"Hoogste diepte = {int(best_depth)} en totale waarschijnlijkheid = {best_total:.2f}.")
    else:
        print("\nConclusie: Geen geldige diepte-informatie beschikbaar in de top 5 classificaties.")
