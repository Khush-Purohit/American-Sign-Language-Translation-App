import pickle
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque


LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Load model and scaler
model = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# Smoothing buffer
SMOOTH_WINDOW = 5
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)
CONFIDENCE_THRESHOLD = 0.4

def normalize_landmarks_3d(hand_landmarks):
    """Same normalization as training - WITH z-coordinate"""
    data_aux = []
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    z_coords = [lm.z for lm in hand_landmarks.landmark]
    
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = max_z - min_z
    span = max(span_x, span_y, span_z)
    
    if span < 1e-6:
        span = 1.0
    
    for landmark in hand_landmarks.landmark:
        data_aux.append((landmark.x - wrist_x) / span)
        data_aux.append((landmark.y - wrist_y) / span)
        data_aux.append((landmark.z - wrist_z) / span)
    
    return data_aux

cap = cv2.VideoCapture(0)

print("Real-time ASL Recognition (with 3D landmarks)")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    display_text = "No Hand Detected"
    predicted_letter = "?"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Normalize and predict
            data_aux = normalize_landmarks_3d(hand_landmarks)
            data_scaled = scaler.transform([np.array(data_aux)])
            
            # Get prediction probabilities
            prediction_proba = model.predict_proba(data_scaled)[0]
            prediction_buffer.append(prediction_proba)
            
            # Average predictions over buffer (smoothing)
            avg_proba = np.mean(prediction_buffer, axis=0)
            predicted_idx = np.argmax(avg_proba)
            confidence = avg_proba[predicted_idx]
            
            # Get predicted class (numeric)
            predicted_class = model.classes_[predicted_idx]
            
            # Convert to letter - ALWAYS convert numeric to letter
            predicted_letter = LABEL_MAP.get(int(predicted_class), '?')
            
            # Only show prediction if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{predicted_letter} ({confidence*100:.1f}%)"
            else:
                display_text = "Uncertain"
                predicted_letter = "?"

            # Draw bounding box
            x_coords = [lm.x * W for lm in hand_landmarks.landmark]
            y_coords = [lm.y * H for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords)) - 20, int(min(y_coords)) - 20
            x2, y2 = int(max(x_coords)) + 20, int(max(y_coords)) + 20
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display letter on bounding box
            cv2.putText(frame, predicted_letter, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display overall status
    cv2.putText(frame, display_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.putText(frame, "Press 'Q' to Quit", (10, H - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()