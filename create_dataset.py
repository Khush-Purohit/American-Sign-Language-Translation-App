import os
import mediapipe as mp
import cv2
import pickle
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

def normalize_landmarks_3d(hand_landmarks):
    """Normalize landmarks to be position and scale invariant (with z-coordinate)"""
    data_aux = []
    
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    z_coords = [lm.z for lm in hand_landmarks.landmark]
    
    
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    span = max(span_x, span_y)
    

    if span < 1e-6:
        span = 1.0
    

    for landmark in hand_landmarks.landmark:
        normalized_x = (landmark.x - wrist_x) / span
        normalized_y = (landmark.y - wrist_y) / span
        normalized_z = (landmark.z - wrist_z) / span  
        
        data_aux.append(normalized_x)
        data_aux.append(normalized_y)
        data_aux.append(normalized_z)  
    
    return data_aux

print("Processing dataset...")
processed_count = 0
skipped_count = 0

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    

    if not os.path.isdir(dir_path) or dir_.startswith('.'):
        continue
    
    print(f"Processing class: {dir_}")
    
    for img_path in os.listdir(dir_path):
    
        if img_path.startswith('.'):
            continue
            
        full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(full_path)
        
        if img is None:
            skipped_count += 1
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                normalized_data = normalize_landmarks_3d(hand_landmarks)
                data.append(normalized_data)
                labels.append(dir_)
                processed_count += 1
        else:
            skipped_count += 1


hands.close()


with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\nSuccessfully processed {processed_count} samples")
print(f"Skipped {skipped_count} images (no hand detected or unreadable)")
print(f"Total classes: {len(set(labels))}")
print(f"Classes: {sorted(set(labels))}")
print(f"Data saved to data.pickle")
print(f"Feature dimension: {len(data[0]) if data else 0} (21 landmarks × 3 coordinates = 63 features)")